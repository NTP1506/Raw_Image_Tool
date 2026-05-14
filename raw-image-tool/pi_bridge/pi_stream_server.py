import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Tuple

import cv2
import numpy as np


def _parse_bayer_order(sensor_format: str) -> str:
    text = (sensor_format or "").upper()
    for token in ("RGGB", "BGGR", "GRBG", "GBRG"):
        if token in text:
            return token
    return "BGGR"


def _to_u8(raw: np.ndarray) -> np.ndarray:
    if raw.dtype == np.uint8:
        return raw
    raw_f = raw.astype(np.float32)
    p1, p99 = np.percentile(raw_f, [1, 99])
    scaled = (raw_f - p1) / max(p99 - p1, 1e-6)
    scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def _demosaic_preview(raw: np.ndarray, bayer_order: str) -> np.ndarray:
    raw_u8 = _to_u8(raw)
    code_map = {
        "BGGR": cv2.COLOR_BayerBG2BGR,
        "RGGB": cv2.COLOR_BayerRG2BGR,
        "GRBG": cv2.COLOR_BayerGR2BGR,
        "GBRG": cv2.COLOR_BayerGB2BGR,
    }
    code = code_map.get(bayer_order, cv2.COLOR_BayerBG2BGR)
    return cv2.cvtColor(raw_u8, code)


class PiCaptureBridge:
    def __init__(self, width: int = 1280, height: int = 720) -> None:
        try:
            from picamera2 import Picamera2
        except Exception as exc:
            raise RuntimeError(
                "picamera2 is required on Raspberry Pi. Install with: sudo apt install python3-picamera2"
            ) from exc

        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            raw={"size": self.picam2.sensor_resolution},
            buffer_count=4,
        )
        self.picam2.configure(config)
        self.picam2.start()

    def capture_pair(self) -> Tuple[bytes, bytes, bytes]:
        request = self.picam2.capture_request()
        try:
            main_rgb = request.make_array("main")
            raw_arr = request.make_array("raw")
            metadata: Dict = request.get_metadata() or {}
        finally:
            request.release()

        main_bgr = cv2.cvtColor(main_rgb, cv2.COLOR_RGB2BGR)
        ok, jpg = cv2.imencode(".jpg", main_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise RuntimeError("Failed to encode ISP frame as JPEG")

        sensor_format = str(metadata.get("SensorFormat", ""))
        bayer_order = _parse_bayer_order(sensor_format)

        raw_bytes = io.BytesIO()
        np.savez_compressed(
            raw_bytes,
            raw=raw_arr,
            bayer_order=np.array(bayer_order),
            sensor_format=np.array(sensor_format),
        )

        preview_bgr = _demosaic_preview(raw_arr, bayer_order)
        ok2, preview_jpg = cv2.imencode(".jpg", preview_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok2:
            raise RuntimeError("Failed to encode RAW preview as JPEG")

        return jpg.tobytes(), raw_bytes.getvalue(), preview_jpg.tobytes()


class Handler(BaseHTTPRequestHandler):
    bridge: PiCaptureBridge = None

    def _send_bytes(self, payload: bytes, content_type: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, data: Dict, code: int = 200) -> None:
        payload = json.dumps(data).encode("utf-8")
        self._send_bytes(payload, "application/json", code=code)

    def do_GET(self) -> None:
        try:
            if self.path == "/health":
                self._send_json({"ok": True})
                return

            if self.path in ("/frame/isp.jpg", "/frame/raw.npz", "/frame/raw_preview.jpg"):
                isp_jpg, raw_npz, raw_preview_jpg = self.bridge.capture_pair()
                if self.path == "/frame/isp.jpg":
                    self._send_bytes(isp_jpg, "image/jpeg")
                elif self.path == "/frame/raw.npz":
                    self._send_bytes(raw_npz, "application/octet-stream")
                else:
                    self._send_bytes(raw_preview_jpg, "image/jpeg")
                return

            self._send_json({"error": "Not found"}, code=404)
        except Exception as exc:
            self._send_json({"error": str(exc)}, code=500)


def main() -> None:
    bridge = PiCaptureBridge(width=1280, height=720)
    Handler.bridge = bridge
    server = ThreadingHTTPServer(("0.0.0.0", 8088), Handler)
    print("Pi stream server listening on http://0.0.0.0:8088")
    print("Endpoints: /health, /frame/isp.jpg, /frame/raw.npz, /frame/raw_preview.jpg")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import ImageTk

from shared.constants import (
    FILETYPES,
    PI_PREVIEW_JPEG_QUALITY,
    PREVIEW_MAX_H,
    PREVIEW_MAX_W,
)
from shared.helpers import bgr_to_photoimage, ensure_outputs_dir
from raw.raw_loader import load_raw_image
from raw.pi_processing import simulate_libcamera_processing, simulate_pi_jpeg_quality
from processing import (
    apply_gamma,
    apply_resize,
    auto_white_balance,
    multi_scale_sharpen,
    recover_shadow_details,
    hdr_tone_mapping,
    denoise_bilateral_adaptive,
    apply_clarity,
    apply_tone_curve,
    enhance_object_edges,
)


class MainWindow:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("RAW Image Processing Tool")
        self.root.resizable(True, True)

        # State
        self._original_bgr: np.ndarray | None = None
        self._pi_processed_bgr: np.ndarray | None = None  # After Pi pipeline
        self._processed_bgr: np.ndarray | None = None
        self._orig_photo: ImageTk.PhotoImage | None = None
        self._pi_photo: ImageTk.PhotoImage | None = None
        self._proc_photo: ImageTk.PhotoImage | None = None

        self._build_ui()

    # UI construction
    def _build_ui(self):
        root = self.root
        root.configure(bg="#1e1e1e")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame",       background="#1e1e1e")
        style.configure("TLabel",       background="#1e1e1e", foreground="#d4d4d4", font=("Segoe UI", 10))
        style.configure("TButton",      font=("Segoe UI", 10), padding=6)
        style.configure("TCheckbutton", background="#1e1e1e", foreground="#d4d4d4", font=("Segoe UI", 10))
        style.configure("TScale",       background="#1e1e1e")
        style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"), background="#1e1e1e", foreground="#9cdcfe")

        # Top bar
        top = ttk.Frame(root, padding=8)
        top.pack(fill=tk.X)

        # File input controls
        ttk.Button(top, text="Open Image", command=self._open_image).pack(side=tk.LEFT, padx=4)
        self._file_label = ttk.Label(top, text="No file loaded", foreground="#858585")
        self._file_label.pack(side=tk.LEFT, padx=10)

        # Preview pane
        preview_frame = ttk.Frame(root, padding=(8, 0))
        preview_frame.pack(fill=tk.BOTH, expand=True)

        # Original
        orig_col = ttk.Frame(preview_frame)
        orig_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        ttk.Label(orig_col, text="Original", style="Header.TLabel").pack()
        self._orig_canvas = tk.Canvas(orig_col, bg="#2d2d2d", width=PREVIEW_MAX_W, height=PREVIEW_MAX_H,
                                       highlightthickness=1, highlightbackground="#3e3e3e")
        self._orig_canvas.pack(fill=tk.BOTH, expand=True)

        # Pi Image
        pi_col = ttk.Frame(preview_frame)
        pi_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        ttk.Label(pi_col, text="Pi Image", style="Header.TLabel").pack()
        self._pi_canvas = tk.Canvas(pi_col, bg="#2d2d2d", width=PREVIEW_MAX_W, height=PREVIEW_MAX_H,
                                     highlightthickness=1, highlightbackground="#3e3e3e")
        self._pi_canvas.pack(fill=tk.BOTH, expand=True)

        # Processed
        proc_col = ttk.Frame(preview_frame)
        proc_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        ttk.Label(proc_col, text="Processed", style="Header.TLabel").pack()
        self._proc_canvas = tk.Canvas(proc_col, bg="#2d2d2d", width=PREVIEW_MAX_W, height=PREVIEW_MAX_H,
                                       highlightthickness=1, highlightbackground="#3e3e3e")
        self._proc_canvas.pack(fill=tk.BOTH, expand=True)

        # Controls
        ctrl = ttk.Frame(root, padding=12)
        ctrl.pack(fill=tk.X)

        self._build_controls(ctrl)

        # Action buttons
        btn_row = ttk.Frame(root, padding=(12, 4, 12, 12))
        btn_row.pack(fill=tk.X)

        ttk.Button(btn_row, text="Pi Image",     command=self._apply_pi_pipeline).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_row, text="Process",     command=self._process).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_row, text="Reset",       command=self._reset).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_row, text="Save Output", command=self._save).pack(side=tk.RIGHT, padx=6)

    def _build_controls(self, parent: ttk.Frame):
        # Gamma
        row_gamma = ttk.Frame(parent)
        row_gamma.pack(fill=tk.X, pady=3)

        self._gamma_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row_gamma, text="Gamma", variable=self._gamma_var, width=14).pack(side=tk.LEFT)

        ttk.Label(row_gamma, text="Gamma value").pack(side=tk.LEFT, padx=(10, 2))
        self._gamma_val = tk.DoubleVar(value=1.0)
        ttk.Scale(row_gamma, from_=0.1, to=5.0, variable=self._gamma_val, orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT)
        ttk.Label(row_gamma, textvariable=self._gamma_val, width=4).pack(side=tk.LEFT)

        # Resize
        row_resize = ttk.Frame(parent)
        row_resize.pack(fill=tk.X, pady=3)

        self._resize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row_resize, text="Resize", variable=self._resize_var, width=14).pack(side=tk.LEFT)

        ttk.Label(row_resize, text="Scale (%)").pack(side=tk.LEFT, padx=(10, 2))
        self._resize_scale = tk.IntVar(value=100)
        ttk.Scale(row_resize, from_=10, to=200, variable=self._resize_scale, orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT)
        ttk.Label(row_resize, textvariable=self._resize_scale, width=4).pack(side=tk.LEFT)

        # ── Advanced Enhancements ──────────────────────────────────
        sep = ttk.Frame(parent, height=2, relief=tk.SUNKEN)
        sep.pack(fill=tk.X, pady=6, padx=0)

        # Auto Denoise
        row_denoise = ttk.Frame(parent)
        row_denoise.pack(fill=tk.X, pady=3)

        self._denoise_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row_denoise, text="Auto Denoise", variable=self._denoise_var, width=14).pack(side=tk.LEFT)

        ttk.Label(row_denoise, text="Strength").pack(side=tk.LEFT, padx=(10, 2))
        self._denoise_strength = tk.DoubleVar(value=0.26)
        ttk.Scale(row_denoise, from_=0.1, to=2.0, variable=self._denoise_strength, orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT)
        ttk.Label(row_denoise, textvariable=self._denoise_strength, width=4).pack(side=tk.LEFT)

        # Detail Recovery
        row_detail = ttk.Frame(parent)
        row_detail.pack(fill=tk.X, pady=3)

        self._detail_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row_detail, text="Detail Recover", variable=self._detail_var, width=14).pack(side=tk.LEFT)

        ttk.Label(row_detail, text="Strength").pack(side=tk.LEFT, padx=(10, 2))
        self._detail_strength = tk.DoubleVar(value=0.68)
        ttk.Scale(row_detail, from_=0.0, to=2.0, variable=self._detail_strength, orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT)
        ttk.Label(row_detail, textvariable=self._detail_strength, width=4).pack(side=tk.LEFT)

        # Auto White Balance
        row_awb = ttk.Frame(parent)
        row_awb.pack(fill=tk.X, pady=3)

        self._awb_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row_awb, text="Auto WB", variable=self._awb_var, width=14).pack(side=tk.LEFT)

        ttk.Label(row_awb, text="Strength").pack(side=tk.LEFT, padx=(10, 2))
        self._awb_strength = tk.DoubleVar(value=0.18)
        ttk.Scale(row_awb, from_=0.0, to=1.0, variable=self._awb_strength, orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT)
        ttk.Label(row_awb, textvariable=self._awb_strength, width=4).pack(side=tk.LEFT)

        # HDR Tone Mapping
        row_hdr = ttk.Frame(parent)
        row_hdr.pack(fill=tk.X, pady=3)

        self._hdr_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row_hdr, text="HDR Tone Map", variable=self._hdr_var, width=14).pack(side=tk.LEFT)

        ttk.Label(row_hdr, text="Strength").pack(side=tk.LEFT, padx=(10, 2))
        self._hdr_strength = tk.DoubleVar(value=0.72)
        ttk.Scale(row_hdr, from_=0.0, to=2.0, variable=self._hdr_strength, orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT)
        ttk.Label(row_hdr, textvariable=self._hdr_strength, width=4).pack(side=tk.LEFT)

        # Multi-scale Sharpen
        row_ms_sharpen = ttk.Frame(parent)
        row_ms_sharpen.pack(fill=tk.X, pady=3)

        self._ms_sharpen_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row_ms_sharpen, text="Smart Sharpen", variable=self._ms_sharpen_var, width=14).pack(side=tk.LEFT)

        ttk.Label(row_ms_sharpen, text="Strength").pack(side=tk.LEFT, padx=(10, 2))
        self._ms_sharpen_strength = tk.DoubleVar(value=4.5)
        ttk.Scale(row_ms_sharpen, from_=0.0, to=5.0, variable=self._ms_sharpen_strength, orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT)
        ttk.Label(row_ms_sharpen, textvariable=self._ms_sharpen_strength, width=4).pack(side=tk.LEFT)

        # Clarity
        row_clarity = ttk.Frame(parent)
        row_clarity.pack(fill=tk.X, pady=3)

        self._clarity_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row_clarity, text="Clarity", variable=self._clarity_var, width=14).pack(side=tk.LEFT)

        ttk.Label(row_clarity, text="Strength").pack(side=tk.LEFT, padx=(10, 2))
        self._clarity_strength = tk.DoubleVar(value=1.20)
        ttk.Scale(row_clarity, from_=0.0, to=2.0, variable=self._clarity_strength, orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT)
        ttk.Label(row_clarity, textvariable=self._clarity_strength, width=4).pack(side=tk.LEFT)

        # Tone Curve
        row_tone = ttk.Frame(parent)
        row_tone.pack(fill=tk.X, pady=3)

        self._tone_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row_tone, text="Tone Curve", variable=self._tone_var, width=14).pack(side=tk.LEFT)

        ttk.Label(row_tone, text="Strength").pack(side=tk.LEFT, padx=(10, 2))
        self._tone_strength = tk.DoubleVar(value=0.25)
        ttk.Scale(row_tone, from_=0.0, to=1.0, variable=self._tone_strength, orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT)
        ttk.Label(row_tone, textvariable=self._tone_strength, width=4).pack(side=tk.LEFT)

        # Object Edge Boost (for detection tasks)
        row_obj_edge = ttk.Frame(parent)
        row_obj_edge.pack(fill=tk.X, pady=3)

        self._obj_edge_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row_obj_edge, text="Object Edge", variable=self._obj_edge_var, width=14).pack(side=tk.LEFT)

        ttk.Label(row_obj_edge, text="Strength").pack(side=tk.LEFT, padx=(10, 2))
        self._obj_edge_strength = tk.DoubleVar(value=1.45)
        ttk.Scale(row_obj_edge, from_=0.0, to=2.0, variable=self._obj_edge_strength, orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT)
        ttk.Label(row_obj_edge, textvariable=self._obj_edge_strength, width=4).pack(side=tk.LEFT)

    # Actions
    def _apply_pi_pipeline(self):
        if self._original_bgr is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        try:
            # Apply local libcamera-like processing pipeline
            self._pi_processed_bgr = simulate_libcamera_processing(
                self._original_bgr,
                apply_demosaic=True,
                apply_white_balance=True,
                apply_gamma=True,
            )

            # Add JPEG compression simulation
            self._pi_processed_bgr = simulate_pi_jpeg_quality(
                self._pi_processed_bgr,
                quality=PI_PREVIEW_JPEG_QUALITY,
            )

            # Show on Pi canvas
            self._show_pi_image(self._pi_processed_bgr)

        except Exception as exc:
            messagebox.showerror("Pi Pipeline Error", str(exc))
            return

    def _open_image(self):
        path = filedialog.askopenfilename(
            title="Open Image",
            filetypes=FILETYPES,
        )
        if not path:
            return

        try:
            self._original_bgr = load_raw_image(path)
        except Exception as exc:
            messagebox.showerror("Load Error", str(exc))
            return

        self._processed_bgr = None
        self._pi_processed_bgr = None
        self._file_label.config(text=os.path.basename(path))
        self._show_original(self._original_bgr)
        self._pi_canvas.delete("all")  # Clear Pi canvas
        self._clear_processed_canvas()  # Clear Processed canvas

    def _process(self):
        if self._original_bgr is None:
            messagebox.showwarning("No Image", "Please open an image first.")
            return

        # Use Pi-processed image if available, otherwise use original
        base_img = self._original_bgr
        img = base_img.copy()

        # Adaptive boost for low-light scenes to improve object recall.
        gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
        gray_f = gray.astype(np.float32) / 255.0

        # Scene darkness from mid-tone percentile (data-driven per image).
        p60_luma = float(np.percentile(gray_f, 60))
        low_light_ratio = float(np.clip(1.0 - p60_luma, 0.0, 1.0))

        # Noise estimate from robust high-frequency energy (MAD of Laplacian).
        lap = cv2.Laplacian(gray_f, cv2.CV_32F, ksize=3)
        lap_abs = np.abs(lap)
        lap_mad = float(np.median(np.abs(lap_abs - np.median(lap_abs))))
        lap_p95 = float(np.percentile(lap_abs, 95)) + 1e-6
        noise_ratio = float(np.clip(lap_mad / lap_p95, 0.0, 1.0))

        sobel_x = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.hypot(sobel_x, sobel_y)
        grad_p90 = float(np.percentile(grad_mag, 90))
        grad_p99 = float(np.percentile(grad_mag, 99)) + 1e-6
        texture_ratio = float(np.clip(grad_p90 / grad_p99, 0.0, 1.0))

        # Bright highlight occupancy from dynamic threshold (95th percentile).
        p95_luma = float(np.percentile(gray_f, 95))
        highlight_ratio = float(np.mean(gray_f >= p95_luma))

        bgr_mean = np.mean(base_img.astype(np.float32), axis=(0, 1)) / 255.0
        avg_b, avg_g, avg_r = bgr_mean
        color_cast_metric = float(np.std([avg_b, avg_g, avg_r]) / (np.mean([avg_b, avg_g, avg_r]) + 1e-6))
        color_cast_ratio = float(np.clip(color_cast_metric, 0.0, 1.0))

        try:
            # Auto Enhancements
            if self._denoise_var.get():
                denoise_scale = 1.0 + low_light_ratio * noise_ratio * (1.0 - texture_ratio)
                denoise_strength = min(2.0, self._denoise_strength.get() * denoise_scale)
                img = denoise_bilateral_adaptive(img, strength=denoise_strength)

            if self._detail_var.get():
                detail_scale = 1.0 + low_light_ratio * noise_ratio * (1.0 - highlight_ratio)
                detail_strength = min(2.0, self._detail_strength.get() * detail_scale)
                img = recover_shadow_details(img, strength=detail_strength)

            if self._awb_var.get():
                awb_scale = 1.0 + color_cast_ratio * low_light_ratio
                awb_strength = min(1.0, self._awb_strength.get() * awb_scale)
                img = auto_white_balance(img, strength=awb_strength)

            if self._hdr_var.get():
                hdr_scale = 1.0 + low_light_ratio * highlight_ratio
                hdr_strength = min(2.0, self._hdr_strength.get() * hdr_scale)
                img = hdr_tone_mapping(img, strength=hdr_strength)

            # Standard Effects
            if self._ms_sharpen_var.get():
                sharpen_scale = 1.0 + low_light_ratio * texture_ratio * (1.0 - noise_ratio)
                sharpen_strength = min(5.0, self._ms_sharpen_strength.get() * sharpen_scale)
                img = multi_scale_sharpen(img, strength=sharpen_strength)

            if self._clarity_var.get():
                clarity_scale = 1.0 + low_light_ratio * texture_ratio * (1.0 - noise_ratio)
                clarity_strength = min(2.0, self._clarity_strength.get() * clarity_scale)
                img = apply_clarity(img, strength=clarity_strength)

            if self._tone_var.get():
                img = apply_tone_curve(img, strength=self._tone_strength.get())

            if self._obj_edge_var.get():
                obj_edge_scale = 1.0 + low_light_ratio * texture_ratio * (1.0 - noise_ratio)
                obj_edge_strength = min(2.0, self._obj_edge_strength.get() * obj_edge_scale)
                img = enhance_object_edges(img, strength=obj_edge_strength)

            if self._gamma_var.get():
                img = apply_gamma(img, gamma=round(self._gamma_val.get(), 2))

            if self._resize_var.get():
                scale = self._resize_scale.get() / 100.0
                img = apply_resize(img, scale=scale)

        except Exception as exc:
            messagebox.showerror("Processing Error", str(exc))
            return

        self._processed_bgr = img
        self._show_processed(img)

    def _reset(self):
        self._original_bgr = None
        self._pi_processed_bgr = None
        self._processed_bgr = None
        self._orig_canvas.delete("all")
        self._pi_canvas.delete("all")
        self._proc_canvas.delete("all")

    def _save(self):
        # Prioritize: Processed > Pi > None
        save_img = self._processed_bgr if self._processed_bgr is not None else self._pi_processed_bgr
        
        if save_img is None:
            messagebox.showwarning("Nothing to Save", "Process an image or apply Pi pipeline first.")
            return

        path = filedialog.asksaveasfilename(
            title="Save Output",
            defaultextension=".png",
            initialdir=ensure_outputs_dir(),
            filetypes=[
                ("PNG image",  "*.png"),
                ("JPEG image", "*.jpg"),
                ("BMP image",  "*.bmp"),
                ("TIFF image", "*.tiff"),
            ],
        )
        if not path:
            return

        ok = cv2.imwrite(path, save_img)
        if ok:
            messagebox.showinfo("Saved", f"Image saved to:\n{path}")
        else:
            messagebox.showerror("Save Error", f"Failed to write:\n{path}")

    # Canvas helpers
    def _show_original(self, bgr: np.ndarray):
        self._orig_photo = bgr_to_photoimage(bgr, PREVIEW_MAX_W, PREVIEW_MAX_H)
        self._draw_on_canvas(self._orig_canvas, self._orig_photo)

    def _show_pi_image(self, bgr: np.ndarray):
        self._pi_photo = bgr_to_photoimage(bgr, PREVIEW_MAX_W, PREVIEW_MAX_H)
        self._draw_on_canvas(self._pi_canvas, self._pi_photo)

    def _show_processed(self, bgr: np.ndarray):
        self._proc_photo = bgr_to_photoimage(bgr, PREVIEW_MAX_W, PREVIEW_MAX_H)
        self._draw_on_canvas(self._proc_canvas, self._proc_photo)

    def _draw_on_canvas(self, canvas: tk.Canvas, photo: ImageTk.PhotoImage):
        canvas.delete("all")
        cw = canvas.winfo_width()  or PREVIEW_MAX_W
        ch = canvas.winfo_height() or PREVIEW_MAX_H
        canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=photo)

    def _clear_processed_canvas(self):
        self._proc_canvas.delete("all")

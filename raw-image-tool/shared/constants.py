PREVIEW_MAX_W = 400
PREVIEW_MAX_H = 500

FILETYPES = [
    (
        "All supported images",
        "*.dng *.cr2 *.nef *.arw *.raf *.rw2 *.orf *.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp",
    ),
    ("RAW files", "*.dng *.cr2 *.nef *.arw *.raf *.rw2 *.orf"),
    ("Standard images", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp"),
    ("All files", "*.*"),
]

SUPPORTED_RAW = {".dng", ".cr2", ".nef", ".arw", ".raf", ".rw2", ".orf"}
SUPPORTED_STANDARD = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

PI_GAMMA = 0.5
PI_WHITE_BALANCE_SCALE = 0.95
DEFAULT_PI_JPEG_QUALITY = 85
PI_PREVIEW_JPEG_QUALITY = 90

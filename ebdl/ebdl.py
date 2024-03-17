from PIL import Image
import numpy as np


def preprocess(img: Image.Image, wtm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if img.mode != "L":
        print("Warning: this function is made for grayscale images")
        img = img.convert(mode="L")

    if wtm.ndim != 2:
        print("Error: watermark should have dimension of 2")
        raise

    raw = np.array(img)
    # print(f"Image shape: {raw.shape}")
    # print(f"Watermark shape: {wtm.shape}")

    if wtm.shape[0] < raw.shape[0] or wtm.shape[1] < raw.shape[1]:
        print("Error: watermark should not smaller than image")
        raise
    wtm = wtm.astype(float)
    wtm.resize(raw.shape)

    return [raw, wtm]


def embed(
    img: Image.Image, wtm: np.ndarray, msg: bool = True, alpha: float = 1
) -> Image.Image:
    raw, wtm = preprocess(img, wtm)
    raw = raw + alpha * (1 if msg else -1) * wtm
    raw = np.around(raw).clip(0, 255).astype(np.uint8)
    return Image.fromarray(raw, mode="L")


def detect(img: Image.Image, wtm: np.ndarray) -> float:
    raw, wtm = preprocess(img, wtm)
    res = raw * wtm
    return np.sum(res, dtype=float) / raw.size

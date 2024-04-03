from PIL import Image
import numpy as np
import math


def make_watermark(seed, size: tuple[int, int]) -> np.ndarray:
    np.random.seed(seed)
    wtm = np.random.randint(0, 256, size=size).astype(float)
    wtm -= np.mean(wtm)
    wtm /= np.std(wtm)
    return wtm


def embed(img: Image.Image, wtm_list: list[np.ndarray], msg: int) -> Image.Image:
    if img.mode != "L":
        print("Warning: this function is made for grayscale images")
        img = img.convert(mode="L")

    wtm_threshold = 2 ** len(wtm_list) - 1

    if wtm_threshold < msg:
        print(
            f"Error: Too few watermarks: {len(wtm_list)} representing [0, {wtm_threshold}) < {msg}"
        )
        raise

    raw = np.array(img)
    wtm_embed = [
        wtm_list[i] if (msg >> i) & 1 else -wtm_list[i] for i in range(len(wtm_list))
    ]
    for i in range(len(wtm_embed)):
        wtm_embed[i] = wtm_embed[i].astype(float)
        wtm_embed[i].resize(raw.shape)

    wtm_sum = np.sum(wtm_embed, axis=0)
    wtm_sum = wtm_sum / wtm_sum.std() * math.sqrt(8)

    raw = raw + wtm_sum
    raw = np.around(raw).clip(0, 255).astype(np.uint8)
    return Image.fromarray(raw, mode="L")


def detect(img: Image.Image, wtm_list: list[np.ndarray]) -> tuple[int, float]:
    if img.mode != "L":
        print("Warning: this function is made for grayscale images")
        img = img.convert(mode="L")

    raw = np.array(img)
    msg = 0
    confidence = 0
    for i in range(len(wtm_list)):
        wtm = wtm_list[i].astype(float)
        wtm.resize(raw.shape)
        val = np.mean(raw * wtm)
        msg += (1 if val > 0 else 0) << i
        confidence += abs(val) - 0.50

    confidence /= len(wtm_list)
    return [msg, confidence * 2]

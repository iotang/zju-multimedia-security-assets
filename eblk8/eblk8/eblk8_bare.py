from PIL import Image
import numpy as np
import math


def make_watermark(seed, size: tuple[int, int]) -> np.ndarray:
    np.random.seed(seed)
    wtm = np.random.normal(0, 1, size=size).astype(float)
    wtm -= np.mean(wtm)
    wtm /= np.std(wtm)
    return wtm


def make_watermark_2(seed, size: tuple[int, int]) -> np.ndarray:
    np.random.seed(seed)
    wtm = np.random.randint(0, 256, size=size).astype(float)
    wtm -= np.mean(wtm)
    wtm /= np.std(wtm)
    return wtm


def embed(
    img: Image.Image, msg: int, length: int, seed: int, strength: float = -1
) -> Image.Image:
    if img.mode != "L":
        print("Warning: this function is made for grayscale images")
        img = img.convert(mode="L")

    wtm_threshold = 2**length - 1

    if wtm_threshold < msg:
        print(
            f"Error: Too few watermarks: {length} representing [0, {wtm_threshold}) < {msg}"
        )
        raise

    raw = np.array(img).astype(float)
    wtm_list = [make_watermark(seed + i, (8, 8)) for i in range(length)]
    wtm_embed = [wtm_list[i] if (msg >> i) & 1 else -wtm_list[i] for i in range(length)]

    wtm_sum = np.sum(wtm_embed, axis=0)
    wtm_sum = (
        wtm_sum / wtm_sum.std() * (math.sqrt(length) if strength < 0 else strength)
    )

    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            raw[i][j] += wtm_sum[i & 7][j & 7]
    raw = np.around(raw).clip(0, 255).astype(np.uint8)
    return Image.fromarray(raw, mode="L")


def z_cc(
    a: np.ndarray, b: np.ndarray, skip_mean_a: bool = False, skip_mean_b: bool = False
):
    if not skip_mean_a:
        a = a - np.mean(a)
    if not skip_mean_b:
        b = b - np.mean(b)
    return np.sum(a * b) / math.sqrt(np.sum(a * a) * np.sum(b * b))


def detect(img: Image.Image, length: int, seed: int) -> tuple[int, float]:
    if img.mode != "L":
        print("Warning: this function is made for grayscale images")
        img = img.convert(mode="L")

    raw = np.array(img)

    raw_8 = np.zeros((8, 8)).astype(float)
    for i in range(8):
        for j in range(8):
            raw_8[i][j] = np.mean(raw[i::8, j::8])

    raw_8 -= raw_8.mean()
    wtm_list = [make_watermark(seed + i, (8, 8)) for i in range(length)]

    msg = 0
    for i in range(len(wtm_list)):
        val = np.mean(raw_8 * wtm_list[i])
        msg += (1 if val > 0 else 0) << i

    wtm_embed = [wtm_list[i] if (msg >> i) & 1 else -wtm_list[i] for i in range(length)]
    wtm_sum = np.sum(wtm_embed, axis=0)

    return msg, z_cc(raw_8, wtm_sum, skip_mean_a=True) - 0.50

from PIL import Image
import numpy as np

from . import dct, msgtran


def embed(img: Image.Image, msg: str) -> tuple[Image.Image, int]:
    if img.mode != "L":
        print("Warning: this function is made for grayscale images")
        img = img.convert(mode="L")

    raw = np.array(img)
    size_y, size_x = raw.shape
    block_y, block_x = size_y // 8, size_x // 8
    val = dct.dct(raw)

    embed_code = msgtran.encode(msg)
    embed_len = len(embed_code)
    now_embed = 0
    changed_count = 0

    for i in range(block_y * 8):
        for j in range(block_x * 8):
            if i % 8 == 0 and j % 8 == 0:  # 只修改 AC 值
                continue
            if val[i, j] == 0:
                continue
            if (bool(val[i, j] < 0) ^ bool(val[i, j] % 2)) == bool(
                embed_code[now_embed] == "0"
            ):
                val[i, j] = val[i, j] + (1 if val[i, j] < 0 else -1)
                changed_count += 1
            if val[i, j] != 0:
                now_embed += 1
                if now_embed >= embed_len:
                    break
        if now_embed >= embed_len:
            break

    res = dct.idct(val)
    res = res.clip(0, 255).astype(np.uint8)
    return Image.fromarray(res, mode="L"), changed_count


def detect(img: Image.Image) -> str:
    if img.mode != "L":
        print("Warning: this function is made for grayscale images")
        img = img.convert(mode="L")

    raw = np.array(img)
    size_y, size_x = raw.shape
    block_y, block_x = size_y // 8, size_x // 8
    val = dct.dct(raw)
    embed_code = ""
    embed_len = 32
    now_embed = 0

    for i in range(block_y * 8):
        for j in range(block_x * 8):
            if i % 8 == 0 and j % 8 == 0:
                continue
            if i % 8 == 0 and j % 8 == 0:
                continue
            if val[i, j] == 0:
                continue
            embed_code += "1" if (bool(val[i, j] < 0) ^ bool(val[i, j] % 2)) else "0"
            now_embed += 1
            if now_embed == 32:
                embed_len = int(embed_code, 2) + 32
            if now_embed >= embed_len:
                break
        if now_embed >= embed_len:
            break

    return msgtran.decode(embed_code)

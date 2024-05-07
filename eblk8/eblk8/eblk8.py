from PIL import Image
import hashlib
import numpy as np
import math

# trellis used

TRANSFER = [
    [0, 1],  # 0: a
    [2, 3],  # 1: b
    [4, 5],  # 2: c
    [6, 7],  # 3: d
    [0, 1],  # 4: e
    [2, 3],  # 5: f
    [4, 5],  # 6: g
    [6, 7],  # 7: h
]

EDGEVAL = [
    [0, 1],
    [12, 3],
    [10, 15],
    [8, 5],
    [2, 9],
    [14, 13],
    [4, 11],
    [6, 7],
]


def make_watermark(seed: int, size: tuple[int, int]) -> np.ndarray:
    np.random.seed(seed)
    wtm = np.random.normal(0, 1, size=size).astype(float)
    wtm -= np.mean(wtm)
    wtm /= np.std(wtm)
    return wtm


def trellis_watermark(seed, fr: str, to: str) -> np.ndarray:
    s = str(seed) + "##" + fr + "##" + to
    new_seed = (
        int.from_bytes(hashlib.sha256(s.encode(encoding="utf-8")).digest(), "little")
        % 2**32
    )
    return make_watermark(new_seed, (8, 8))


def state_str(state: int, level: int) -> str:
    return chr(ord("A") + state) + str(level)


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

    raw = np.array(img)
    # trel = ""
    trel_state = 0
    wtm_list = []
    for i in range(length + 2):  # add 2 zeroes
        c = (msg >> i) & 1
        # trel += str(EDGEVAL[trel_state][c]) + "-"
        next_state = TRANSFER[trel_state][c]
        wtm_list.append(
            trellis_watermark(seed, state_str(trel_state, i), state_str(next_state, i))
        )
        trel_state = next_state

    wtm_sum = np.sum(wtm_list, axis=0)
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

    state_confidence = [-math.inf for _ in range(8)]
    state_confidence[0] = 0
    state_path = ["" for _ in range(8)]

    raw_8 = np.zeros((8, 8)).astype(float)
    for i in range(8):
        for j in range(8):
            raw_8[i][j] = np.mean(raw[i::8, j::8])

    raw_8 -= raw_8.mean()

    for idx in range(0, length + 2):
        new_confidence = [-math.inf for _ in range(8)]
        new_path = ["" for _ in range(8)]
        for state in range(8):
            if math.isinf(state_confidence[state]):
                continue
            for x in [0, 1]:
                next_state = TRANSFER[state][x]
                val = z_cc(
                    raw_8,
                    trellis_watermark(
                        seed, state_str(state, idx), state_str(next_state, idx)
                    ),
                    skip_mean_a=True,
                )
                if new_confidence[next_state] < state_confidence[state] + val:
                    new_confidence[next_state] = state_confidence[state] + val
                    new_path[next_state] = state_path[state] + format(x, "b")
        state_confidence = new_confidence
        state_path = new_path

    max_index = 0
    if state_confidence[4] > state_confidence[max_index]:
        max_index = 4

    if math.isinf(state_confidence[max_index]):
        return 0, -math.inf

    msg = int(state_path[max_index][-2::-1], base=2)

    trel_state = 0
    wtm_list = []
    for i in range(length):
        c = (msg >> i) & 1
        next_state = TRANSFER[trel_state][c]
        wtm_list.append(
            trellis_watermark(seed, state_str(trel_state, i), state_str(next_state, i))
        )
        trel_state = next_state
    wtm_sum = np.sum(wtm_list, axis=0)

    return msg, z_cc(raw_8, wtm_sum, skip_mean_a=True) - 0.50

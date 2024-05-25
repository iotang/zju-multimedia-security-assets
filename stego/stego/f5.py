from PIL import Image
import numpy as np

from . import dct, msgtran


class MatrixEmbeddingSolver:
    def __init__(self) -> None:
        self.cover = 0
        self.embed = 0

    def batch_cover(self) -> int:
        return self.cover

    def batch_embed(self) -> int:
        return self.embed

    def advance(self, cov: np.ndarray, msg: np.ndarray) -> int:
        raise

    def find_msg(self, cov: np.ndarray) -> np.ndarray:
        raise


class ThreeForTwoSolver(MatrixEmbeddingSolver):
    # 课本上的三对二码
    # b1 = lsb(x1) ^ lsb(x2), b2 = lsb(x2) ^ lsb(x3)

    def __init__(self) -> None:
        MatrixEmbeddingSolver.__init__(self)
        self.cover = 3
        self.embed = 2

    def advance(self, cov: np.ndarray, msg: np.ndarray) -> int:
        b1 = (cov[0] + cov[1]) & 1
        b2 = (cov[1] + cov[2]) & 1
        if msg[0] == b1 and msg[1] == b2:
            return -1
        elif msg[0] != b1 and msg[1] == b2:
            return 0
        elif msg[0] == b1 and msg[1] != b2:
            return 2
        else:
            return 1

    def find_msg(self, cov: np.ndarray) -> np.ndarray:
        return np.array([(cov[0] + cov[1]) & 1, (cov[1] + cov[2]) & 1])


class HammingSolver(MatrixEmbeddingSolver):
    # (7, 4) 海明码，p = 3, n = 7, k = 4
    # m = [d7, d6, d5, d3]

    def __init__(self) -> None:
        MatrixEmbeddingSolver.__init__(self)
        self.cover = 7
        self.embed = 3
        self.h = np.array(
            [  # 0  1  2  3  4  5  6
                [0, 0, 0, 1, 1, 1, 1],
                [0, 1, 1, 0, 0, 1, 1],
                [1, 0, 1, 0, 1, 0, 1],
            ]
        )

    def advance(self, cov: np.ndarray, msg: np.ndarray) -> int:
        x = np.atleast_2d(cov).T
        mt = np.atleast_2d(msg).T
        diff = ((mt - np.matmul(self.h, x)) % 2 + 2) % 2
        idx = 0
        for i in diff.flatten():
            idx = (idx << 1) + i
        return idx - 1

    def find_msg(self, cov: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(cov).T
        mt = np.matmul(self.h, x) % 2
        return mt.flatten()


def embed(
    img: Image.Image, msg: str, solver: MatrixEmbeddingSolver, dct_shuffle: int = 0
) -> tuple[Image.Image, int]:
    if img.mode != "L":
        print("Warning: this function is made for grayscale images")
        img = img.convert(mode="L")

    raw = np.array(img)
    size_y, size_x = raw.shape
    block_y, block_x = size_y // 8, size_x // 8
    val = dct.dct(raw)

    visit_order = []
    for i in range(block_y):
        for j in range(block_x):
            visit_order.append((i, j))
    if dct_shuffle != 0:
        np.random.seed(dct_shuffle)
        np.random.shuffle(visit_order)

    embed_code = msgtran.encode(msg)
    embed_len = len(embed_code)
    embed_code += "0" * solver.batch_embed()
    now_embed = 0
    changed_count = 0

    candi_val = []
    candi_loc = []

    for bi, bj in visit_order:
        for i in range(bi * 8, bi * 8 + 8):
            for j in range(bj * 8, bj * 8 + 8):
                if i % 8 == 0 and j % 8 == 0:  # 只修改 AC 值
                    continue
                if val[i, j] == 0:
                    continue
                candi_val.append(int(val[i, j] < 0) ^ bool(val[i, j] % 2))
                candi_loc.append((i, j))

                if len(candi_loc) == solver.batch_cover():
                    msg = []
                    for _ in range(solver.batch_embed()):
                        msg.append(int(embed_code[now_embed]))
                        now_embed += 1
                    loc = solver.advance(candi_val, msg)
                    if loc >= 0:
                        changed_count += 1
                        if val[candi_loc[loc]] > 0:
                            val[candi_loc[loc]] -= 1
                        else:
                            val[candi_loc[loc]] += 1
                        if val[candi_loc[loc]] == 0:
                            now_embed -= solver.batch_embed()
                            candi_loc.pop(loc)
                            candi_val.pop(loc)
                            continue
                    candi_loc.clear()
                    candi_val.clear()
                    if now_embed >= embed_len:
                        break
            if now_embed >= embed_len:
                break
        if now_embed >= embed_len:
            break

    res = dct.idct(val)
    res = res.clip(0, 255).astype(np.uint8)
    return Image.fromarray(res, mode="L"), changed_count


def detect(
    img: Image.Image, solver: MatrixEmbeddingSolver, dct_shuffle: int = 0
) -> str:
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

    visit_order = []
    for i in range(block_y):
        for j in range(block_x):
            visit_order.append((i, j))
    if dct_shuffle != 0:
        np.random.seed(dct_shuffle)
        np.random.shuffle(visit_order)

    candi_val = []
    # candi_loc = []

    for bi, bj in visit_order:
        for i in range(bi * 8, bi * 8 + 8):
            for j in range(bj * 8, bj * 8 + 8):
                if i % 8 == 0 and j % 8 == 0:  # 只修改 AC 值
                    continue
                if val[i, j] == 0:
                    continue
                candi_val.append(int(val[i, j] < 0) ^ bool(val[i, j] % 2))
                # candi_loc.append((i, j))
                if len(candi_val) == solver.batch_cover():
                    msg = solver.find_msg(candi_val)
                    for x in range(solver.batch_embed()):
                        embed_code += "1" if msg[x] > 0 else "0"
                        now_embed += 1
                        if now_embed == 32:
                            embed_len = int(embed_code, 2) + 32
                        if now_embed >= embed_len:
                            break
                    candi_val.clear()
                    # candi_loc.clear()

                if now_embed >= embed_len:
                    break
            if now_embed >= embed_len:
                break
        if now_embed >= embed_len:
            break

    return msgtran.decode(embed_code)

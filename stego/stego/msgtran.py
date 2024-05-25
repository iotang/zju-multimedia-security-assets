# 把字符串编码为最前面带一个信息长度的二进制串


def encode(msg: str) -> str:
    bt = msg.encode("utf-8")
    bt_hex = bt.hex()
    bt_bin = bin(int(bt_hex, 16))[2:].zfill(len(bt_hex) * 4)
    len_bin = bin(len(bt_bin))[2:].zfill(32)  # 最多放 2^32 位信息
    return len_bin + bt_bin


def decode(bt: str) -> str:
    bt_len = int(bt[:32], 2)
    return bytes.fromhex(hex(int(bt[32:], 2))[2:].zfill(bt_len // 4)).decode(
        "utf-8", errors="replace"
    )

import PIL.Image as Image
import difflib
from stego import f4, dct_stat


def match_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


s = ""
with open("input.txt", "r", encoding="utf-8") as f:
    ff = f.readlines()
    s = "".join([i for i in ff])

img = Image.open("base.png")
img_embed, changed = f4.embed(img, s)
res = f4.detect(img_embed)
img_embed.save("embed_f4.png")
print(s)
print(match_ratio(res, s), changed)

dct_stat.draw_dct_stat(img)
dct_stat.draw_dct_stat_difference(img, img_embed)

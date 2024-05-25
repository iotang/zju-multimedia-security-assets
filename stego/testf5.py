import PIL.Image as Image
import difflib
from stego import f5, dct_stat


def match_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


s = ""
with open("input.txt", "r", encoding="utf-8") as f:
    ff = f.readlines()
    s = "".join([i for i in ff])

img = Image.open("base.png")
dct_stat.draw_dct_stat(img)

print("ThreeForTwoSolver, not shuffled")
img_embed, changed = f5.embed(img, s, f5.ThreeForTwoSolver())
res = f5.detect(img_embed, f5.ThreeForTwoSolver())
img_embed.save("embed_f5_3f2.png")
print(res)
print(match_ratio(res, s), changed)
dct_stat.draw_dct_stat_difference(img, img_embed)

print("HammingSolver, not shuffled")
img_embed, changed = f5.embed(img, s, f5.HammingSolver())
res = f5.detect(img_embed, f5.HammingSolver())
img_embed.save("embed_f5_hamming.png")
print(res)
print(match_ratio(res, s), changed)
dct_stat.draw_dct_stat_difference(img, img_embed)

print("ThreeForTwoSolver, shuffled")
img_embed, changed = f5.embed(img, s, f5.ThreeForTwoSolver(), 10086)
res = f5.detect(img_embed, f5.ThreeForTwoSolver(), 10086)
img_embed.save("embed_f5_3f2_shuffled.png")
print(res)
print(match_ratio(res, s), changed)
dct_stat.draw_dct_stat_difference(img, img_embed)

print("HammingSolver, shuffled")
img_embed, changed = f5.embed(img, s, f5.HammingSolver(), 1919810)
res = f5.detect(img_embed, f5.HammingSolver(), 1919810)
img_embed.save("embed_f5_hamming_shuffled.png")
print(res)
print(match_ratio(res, s), changed)
dct_stat.draw_dct_stat_difference(img, img_embed)

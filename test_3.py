from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import random

from ebdl import ebdl


def bw_count(img: Image.Image) -> float:
    raw = np.array(img.convert(mode="L")).flatten()
    count = 0
    for i in raw:
        if i < 2 or i > 253:
            count += 1
    return 1.00 * count / raw.size


print("Collecting images")

result_path = "./test3_result"
os.makedirs(result_path, exist_ok=True)

dataset_path = "../data"
dataset = []

for file in os.listdir(dataset_path):
    file_path = os.path.join(dataset_path, file)
    if os.path.isfile(file_path):
        try:
            img = Image.open(file_path)
            dataset.append(img)
        except IOError:
            pass

print("Finding images satisfies limit")
random.shuffle(dataset)
img = None
for i in dataset:
    if bw_count(i) >= 0.50:
        img = i
        break

img.save(os.path.join(result_path, "test_3.png"))
height = img.size[1]
width = img.size[0]

os.makedirs(os.path.join(result_path, "test3_watermark"), exist_ok=True)
wtm_set = []
for i in range(0, 200):
    print(f"Generating watermark {i}")
    wtm_set.append(np.random.normal(0, 1, (height, width)))
    wtm_visual = (wtm_set[-1] + 3) * (256.00 / 6)
    wtm_visual = np.round(wtm_visual).clip(0, 255).astype(np.uint8)
    Image.fromarray(wtm_visual).save(
        os.path.join(result_path, "test3_watermark", f"test3_watermark_{i}.png")
    )

result_pos = []
result_ntr = []
result_neg = []
count = 0
thresh = 0.7
true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
for wtm in wtm_set:
    count += 1
    print(f"Processing {count} / {len(wtm_set)}")
    result_pos.append(ebdl.detect(ebdl.embed(img, wtm, True, 1), wtm))
    result_ntr.append(ebdl.detect(img, wtm))
    result_neg.append(ebdl.detect(ebdl.embed(img, wtm, False, 1), wtm))

    if result_pos[-1] > thresh:
        true_pos += 1
    elif result_pos[-1] < -thresh:
        false_neg += 1

    if result_neg[-1] > thresh:
        false_pos += 1
    elif result_neg[-1] < -thresh:
        true_neg += 1

    if result_ntr[-1] > thresh:
        false_pos += 1
    elif result_ntr[-1] < -thresh:
        false_neg += 1

with open(os.path.join(result_path, "result_3.txt"), mode="w") as f:
    f.write(str(result_pos))
    f.write("\n")
    f.write(str(result_ntr))
    f.write("\n")
    f.write(str(result_neg))
    f.write("\n")
    f.write(f"BW count: {bw_count(img)}\n")
    f.write(f"All: {len(wtm_set)}\n")
    f.write(f"TP: {true_pos}\n")
    f.write(f"TN: {true_neg}\n")
    f.write(f"FP: {false_pos}\n")
    f.write(f"FN: {false_neg}\n")

sns.set_theme()
sns.kdeplot(result_pos, color="b")
sns.kdeplot(result_ntr, color="g")
sns.kdeplot(result_neg, color="r")
plt.xlim(-2, 2)
plt.show()

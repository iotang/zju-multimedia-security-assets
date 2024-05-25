from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from ebdl import ebdl


result_path = "./test1_result"
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

max_height = 0
max_width = 0
for img in dataset:
    max_width = max(max_width, img.size[0])
    max_height = max(max_height, img.size[1])

wtm = np.random.normal(0, 1, (max_height, max_width))  # 正态分布
wtm_visual = (wtm + 3) * (256.00 / 6)
wtm_visual = np.round(wtm_visual).clip(0, 255).astype(np.uint8)
Image.fromarray(wtm_visual).save(os.path.join(result_path, "test1_watermark.png"))
np.save(os.path.join(result_path, "test1_watermark.npy"), wtm)

result_pos = []
result_ntr = []
result_neg = []
count = 0
thresh = 0.7
true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
for img in dataset:
    count += 1
    print(f"Processing {count} / {len(dataset)}")
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

with open(os.path.join(result_path, "result_1.txt"), mode="w") as f:
    f.write(str(result_pos))
    f.write("\n")
    f.write(str(result_ntr))
    f.write("\n")
    f.write(str(result_neg))
    f.write("\n")
    f.write(f"All: {len(dataset)}\n")
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

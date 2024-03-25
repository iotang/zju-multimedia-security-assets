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

result_path = "./test4_result"
os.makedirs(result_path, exist_ok=True)

peer_path = "./test2_result"
img = Image.open(os.path.join(peer_path, "test_2.png"))
wtm_set = []
for file in os.listdir(os.path.join(peer_path, "test2_watermark")):
    file_path = os.path.join(os.path.join(peer_path, "test2_watermark"), file)
    if os.path.isfile(file_path):
        try:
            if os.path.splitext(file_path)[1] == ".npy":
                wtm = np.load(file_path)
                wtm_set.append(wtm)
        except IOError:
            pass


result_pos = []
result_ntr = []
result_neg = []
count = 0
thresh = 0.7
true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
for wtm in wtm_set:
    count += 1
    print(f"Processing {count} / {len(wtm_set)}")
    result_pos.append(ebdl.detect_raw(ebdl.embed_noclip(img, wtm, True, 1), wtm))
    result_ntr.append(ebdl.detect(img, wtm))
    result_neg.append(ebdl.detect_raw(ebdl.embed_noclip(img, wtm, False, 1), wtm))

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

with open(os.path.join(result_path, "result_4.txt"), mode="w") as f:
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

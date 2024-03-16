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
        if i < 2 or i > 254:
            count += 1
    return 1.00 * count / raw.size


dataset_path = "../data"
dataset = []

for file in os.listdir(dataset_path):
    file_path = os.path.join(dataset_path, file)
    if os.path.isfile(file_path):
        try:
            print(f"Find image: {file}")
            img = Image.open(file_path)
            if bw_count(img) <= 0.30:
                dataset.append(img)
        except IOError:
            pass

img = random.choice(dataset)
img.save("test_2.png")
height = img.size[1]
width = img.size[0]

wtm_set = []
for i in range(0, 40):
    print(f"Generating watermark {i}")
    wtm_set.append(np.random.normal(0, 1, (height, width)))

result_pos = []
result_ntr = []
result_neg = []
count = 0
for wtm in wtm_set:
    count += 1
    print(f"Processing {count} / {len(wtm_set)}")
    result_pos.append(ebdl.detect(ebdl.embed(img, wtm, True, 1), wtm))
    result_ntr.append(ebdl.detect(img, wtm))
    result_neg.append(ebdl.detect(ebdl.embed(img, wtm, False, 1), wtm))

with open("result_2.txt", mode="w") as f:
    f.write(str(result_pos))
    f.write("\n")
    f.write(str(result_ntr))
    f.write("\n")
    f.write(str(result_neg))

sns.set_theme()
sns.kdeplot(result_pos, color="b")
sns.kdeplot(result_ntr, color="g")
sns.kdeplot(result_neg, color="r")
plt.xlim(-2, 2)
plt.show()

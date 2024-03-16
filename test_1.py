from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from ebdl import ebdl

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
print(wtm)

result_pos = []
result_ntr = []
result_neg = []
count = 0
for img in dataset:
    count += 1
    print(f"Processing {count} / {len(dataset)}")
    result_pos.append(ebdl.detect(ebdl.embed(img, wtm, True, 1), wtm))
    result_ntr.append(ebdl.detect(img, wtm))
    result_neg.append(ebdl.detect(ebdl.embed(img, wtm, False, 1), wtm))

with open("result_1.txt", mode="w") as f:
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

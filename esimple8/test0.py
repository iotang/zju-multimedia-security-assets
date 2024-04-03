from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import random

from esimple8 import esimple8


result_path = "./test0_result"
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

random.shuffle(dataset)

max_height = 0
max_width = 0
for img in dataset:
    max_width = max(max_width, img.size[0])
    max_height = max(max_height, img.size[1])

wtm_list = [
    esimple8.make_watermark(int(random.random() * (2**32)), (max_height, max_width))
    for i in range(8)
]

img = dataset[0]
msg = random.randint(0, 2**8 - 1)
msg_get, confidence = esimple8.detect(esimple8.embed(img, wtm_list, msg), wtm_list)
print(msg, msg_get, msg == msg_get, confidence)
msg_get, confidence = esimple8.detect(img, wtm_list)
print(msg, msg_get, msg == msg_get, confidence)

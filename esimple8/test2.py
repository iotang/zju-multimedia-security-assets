from PIL import Image
import tqdm
import numpy as np
import os
import random
from functools import partial

from esimple8 import esimple8

test_count = 0
correct = 0
incorrect = 0
true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
confidence_wi = []
confidence_wo = []


def test_img(img, wtm_list, msg: int):
    global test_count, correct, incorrect, true_pos, true_neg, false_pos, false_neg
    if msg >= 0:
        msg_get, confidence = esimple8.detect(
            esimple8.embed(img, wtm_list, msg), wtm_list
        )
        confidence_wi.append(confidence)

        test_count += 1
        if confidence > 0 and msg == msg_get:
            correct += 1
        else:
            incorrect += 1

        if confidence > 0:
            true_pos += 1
        else:
            false_neg += 1

    else:
        msg_get, confidence = esimple8.detect(img, wtm_list)
        confidence_wo.append(confidence)

        if confidence > 0:
            false_pos += 1
        else:
            true_neg += 1
    return


if __name__ == "__main__":
    result_path = "./test2_result"
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

    img = random.choice(dataset)

    img.save(os.path.join(result_path, "test.png"))

    max_height = img.size[1]
    max_width = img.size[0]

    print("Ready for running")

    for i in tqdm.tqdm(range(100)):
        wtm_list = [
            esimple8.make_watermark(
                int(random.random() * (2**32)), (max_height, max_width)
            )
            for i in range(8)
        ]
        test_img_p = partial(test_img, img, wtm_list)
        p = tqdm.tqdm(range(-1, 2**8))
        p.set_description(f"Watermark {i + 1}")
        for j in p:
            test_img_p(j)

    with open(os.path.join(result_path, "result.txt"), mode="w") as f:
        f.write(str(confidence_wi))
        f.write("\n")
        f.write(str(confidence_wo))
        f.write("\n")
        f.write(f"All: {len(dataset)}\n")
        f.write(f"Correct: {correct} / {test_count}\n")
        f.write(f"Incorrect: {incorrect} / {test_count}\n")
        f.write(f"TP: {true_pos} / {test_count}\n")
        f.write(f"TN: {true_neg} / {len(dataset)}\n")
        f.write(f"FP: {false_pos} / {len(dataset)}\n")
        f.write(f"FN: {false_neg} / {test_count}\n")

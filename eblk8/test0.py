from PIL import Image
import numpy as np
import os
import random
from functools import partial

from eblk8 import eblk8

test_count = 0
correct = 0
incorrect = 0
true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
confidence_wi = []
confidence_wo = []


def test_img(img, seed, msg: int):
    global test_count, correct, incorrect, true_pos, true_neg, false_pos, false_neg, confidence_wi, confidence_wo
    if msg >= 0:
        msg_get, confidence = eblk8.detect(eblk8.embed(img, msg, 8, seed), 8, seed)
        confidence_wi.append(confidence)
        print(msg, msg_get, confidence)

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
        msg_get, confidence = eblk8.detect(img, 8, seed)
        confidence_wo.append(confidence)

        if confidence > 0:
            false_pos += 1
        else:
            true_neg += 1
    return


if __name__ == "__main__":
    result_path = "./test0_result"
    os.makedirs(result_path, exist_ok=True)

    dataset_path = "../data"
    dataset = []

    # for file in os.listdir(dataset_path):
    #     file_path = os.path.join(dataset_path, file)
    #     if os.path.isfile(file_path):
    #         try:
    #             img = Image.open(file_path)
    #             dataset.append(img)
    #         except IOError:
    #             pass

    # random.shuffle(dataset)

    for i in range(2**8):
        test_img(Image.open("more_data_12.jpg"), 0, i)

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

    print(f"Correct: {correct} / {test_count}\n")

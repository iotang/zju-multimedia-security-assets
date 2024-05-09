from PIL import Image
import numpy as np
import os
import math
import random
from functools import partial
from multiprocessing import Pool, Manager, freeze_support
import tqdm

from eblk8 import eblk8_bare


def test_img(img, seed, shared, confidence_wi, confidence_wo, lock, msg: int):
    if msg >= 0:
        msg_get, confidence = eblk8_bare.detect(
            eblk8_bare.embed(img, msg, 8, seed), 8, seed
        )

        lock.acquire()
        confidence_wi.put(confidence)
        shared["test_count"] += 1
        if confidence > 0 and msg == msg_get:
            shared["correct"] += 1
        else:
            shared["incorrect"] += 1

        if confidence > 0:
            shared["true_pos"] += 1
        else:
            shared["false_neg"] += 1
        lock.release()

    else:
        msg_get, confidence = eblk8_bare.detect(img, 8, seed)

        lock.acquire()
        confidence_wo.put(confidence)
        if confidence > 0:
            shared["false_pos"] += 1
        else:
            shared["true_neg"] += 1
        lock.release()
    return


if __name__ == "__main__":
    freeze_support()

    result_path = "./test5_result"
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

    wtm_seed = 114514

    print("Ready for running")

    manager = Manager()
    shared = manager.dict()
    shared["test_count"] = 0
    shared["correct"] = 0
    shared["incorrect"] = 0
    shared["true_pos"] = 0
    shared["true_neg"] = 0
    shared["false_pos"] = 0
    shared["false_neg"] = 0
    confidence_wi = manager.Queue()
    confidence_wo = manager.Queue()
    confidence_wi_list = []
    confidence_wo_list = []

    for i in tqdm.tqdm(range(len(dataset))):
        img = dataset[i]
        pool = Pool(processes=16)
        lock = manager.Lock()
        test_img_p = partial(
            test_img, img, wtm_seed, shared, confidence_wi, confidence_wo, lock
        )
        p = tqdm.tqdm(
            pool.imap_unordered(test_img_p, [j for j in range(-1, 2**8)]),
            total=2**8 + 1,
        )
        p.set_description(f"Image {i + 1}")
        for _ in p:
            pass
        pool.close()
        pool.join()

        while not confidence_wi.empty():
            confidence_wi_list.append(confidence_wi.get())
        while not confidence_wo.empty():
            confidence_wo_list.append(confidence_wo.get())

        with open(os.path.join(result_path, "result.txt"), mode="w") as f:
            f.write(str(confidence_wi_list))
            f.write("\n")
            f.write(str(confidence_wo_list))
            f.write("\n")
            f.write(f"All: {i + 1}\n")
            f.write(f"Correct: {shared['correct']} / {shared['test_count']}\n")
            f.write(f"Incorrect: {shared['incorrect']} / {shared['test_count']}\n")
            f.write(f"TP: {shared['true_pos']} / {shared['test_count']}\n")
            f.write(f"TN: {shared['true_neg']} / {i + 1}\n")
            f.write(f"FP: {shared['false_pos']} / {i + 1}\n")
            f.write(f"FN: {shared['false_neg']} / {shared['test_count']}\n")

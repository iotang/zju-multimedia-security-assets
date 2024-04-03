from PIL import Image
import tqdm
import numpy as np
import os
import random
from functools import partial
from multiprocessing import Pool, Manager, freeze_support

from esimple8 import esimple8


def test_img(img, wtm_list, shared, lock, msg: int):
    if msg >= 0:
        msg_get, confidence = esimple8.detect(
            esimple8.embed(img, wtm_list, msg), wtm_list
        )

        lock.acquire()
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
        msg_get, confidence = esimple8.detect(img, wtm_list)

        lock.acquire()
        if confidence > 0:
            shared["false_pos"] += 1
        else:
            shared["true_neg"] += 1
        lock.release()
    return


if __name__ == "__main__":
    freeze_support()

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

    for i in range(len(wtm_list)):
        wtm_visual = (wtm_list[i] + 5) * (256.00 / 10)
        wtm_visual = np.round(wtm_visual).clip(0, 255).astype(np.uint8)
        Image.fromarray(wtm_visual).save(os.path.join(result_path, f"wtm_{i}.jpg"))
        np.save(os.path.join(result_path, f"wtm_{i}.npy"), wtm_list[i])

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

    for i in tqdm.tqdm(range(len(dataset))):
        img = dataset[i]
        pool = Pool(processes=16)
        lock = manager.Lock()
        test_img_p = partial(test_img, img, wtm_list, shared, lock)
        p = tqdm.tqdm(
            pool.imap_unordered(test_img_p, [j for j in range(-1, 2**8)]),
            total=2**8 + 1,
        )
        p.set_description(f"Image {i + 1}")
        for _ in p:
            pass
        pool.close()
        pool.join()

    with open(os.path.join(result_path, "result.txt"), mode="w") as f:
        f.write(f"All: {len(dataset)}\n")
        f.write(f"Correct: {shared['correct']} / {shared['test_count']}\n")
        f.write(f"Incorrect: {shared['incorrect']} / {shared['test_count']}\n")
        f.write(f"TP: {shared['true_pos']} / {shared['test_count']}\n")
        f.write(f"TN: {shared['true_neg']} / {len(dataset)}\n")
        f.write(f"FP: {shared['false_pos']} / {len(dataset)}\n")
        f.write(f"FN: {shared['false_neg']} / {shared['test_count']}\n")

from PIL import Image
import tqdm
import numpy as np
import os
import random
from functools import partial

from esimple8 import esimple8


def test_img(img, wtm_list, msg: int):
    msg_get, confidence = esimple8.detect(esimple8.embed(img, wtm_list, msg), wtm_list)
    return (confidence > 0 and msg == msg_get), confidence


if __name__ == "__main__":
    result_path = "./test3_result"
    os.makedirs(result_path, exist_ok=True)

    # dataset_path = "../data"
    # dataset = []

    # for file in os.listdir(dataset_path):
    #     file_path = os.path.join(dataset_path, file)
    #     if os.path.isfile(file_path):
    #         try:
    #             img = Image.open(file_path)
    #             dataset.append(img)
    #         except IOError:
    #             pass

    # while True:
    #     img = random.choice(dataset)
    #     max_height = img.size[1]
    #     max_width = img.size[0]
    #     if max_height * max_width <= 800 * 800:
    #         break

    # img.save(os.path.join(result_path, "test.png"))
    img = Image.open("house.png")
    max_height = img.size[1]
    max_width = img.size[0]

    print("Ready for running")

    msg_len = []
    correct_p = []
    confidence_p = []

    for i in tqdm.tqdm(range(10, 4001, 10)):
        wtm_list = [
            esimple8.make_watermark(
                int(random.random() * (2**32)), (max_height, max_width)
            )
            for _ in range(i)
        ]
        test_img_p = partial(test_img, img, wtm_list)
        test_msg = [random.randint(0, 2**i - 1) for _ in range(100)]
        p = tqdm.tqdm(test_msg)
        p.set_description(f"Length {i}")

        sum_res = 0
        sum_confidence = 0
        for j in p:
            res, confidence = test_img_p(j)
            sum_res += 1 if res else 0
            sum_confidence += confidence

        msg_len.append(i)
        correct_p.append(sum_res / len(test_msg))
        confidence_p.append(sum_confidence / len(test_msg))

    with open(os.path.join(result_path, "result.txt"), mode="w") as f:
        f.write(str(msg_len))
        f.write("\n")
        f.write(str(correct_p))
        f.write("\n")
        f.write(str(confidence_p))
        f.write("\n")

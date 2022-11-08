from os import path, listdir, abort, mkdir
from uuid import uuid4

import PIL
from tqdm import tqdm

from utils import *

LABELLED_DIR = path.join(path.dirname(path.abspath(__file__)), 'labelled')
SEGMENTED_DIR = path.join(path.dirname(path.abspath(__file__)), 'segmented')
if not path.exists(LABELLED_DIR):
    abort("Missing labelled dir.")
if not path.exists(SEGMENTED_DIR):
    mkdir(SEGMENTED_DIR)


def main():
    succ = 0
    fail = 0
    for file in tqdm(listdir(LABELLED_DIR)):
        try:
            img = Image.open(path.join(LABELLED_DIR, file))
        except PIL.UnidentifiedImageError:
            print(f"Failed to open {file}")
            fail += 1
            continue

        img = img.convert("L")
        table = [0] * 156 + [1] * 100
        img = img.point(table, "1")

        segments = h_split(img)
        label = get_label(file)
        if len(label) != len(segments):
            fail += 1
        else:
            segments = [normalize(v_split(segment)) for segment in segments]
            for pair in zip(label, segments):
                with open(path.join(SEGMENTED_DIR, f"{pair[0]}_{uuid4().hex}.bmp"), mode="wb") as f:
                    pair[1].save(f, "BMP")
            succ += 1

    print(f"SUCC-{succ} FAIL-{fail}")


if __name__ == "__main__":
    main()

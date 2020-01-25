from os import path, listdir, remove, rename, mkdir, abort
from random import randint

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

UNLABELLED_DIR = path.join(path.dirname(path.abspath(__file__)), 'unlabelled')
LABELLED_DIR = path.join(path.dirname(path.abspath(__file__)), 'labelled')
if not path.exists(UNLABELLED_DIR):
    abort("Missing unlabelled dir.")
if not path.exists(LABELLED_DIR):
    mkdir(LABELLED_DIR)


def file_process(fn):
    if fn[-4:] != ".jpg":
        return
    data = open(fn, mode="rb").read(11)
    if data[:4] != b'\xff\xd8\xff\xe0' or data[6:] != b'JFIF\x00':
        return

    mp_img = mpimg.imread(fn)
    plt.ion()
    plt.imshow(mp_img)
    plt.show()
    plt.pause(0.001)
    ans = input(f"> ")
    return ans


def main():
    print("Input 'q' to exit.")
    for file in listdir(UNLABELLED_DIR):
        ans = file_process(path.join(UNLABELLED_DIR, file))
        if not ans:
            remove(path.join(UNLABELLED_DIR, file))
        elif ans != "q":
            rename(path.join(UNLABELLED_DIR, file), path.join(LABELLED_DIR, f"{ans}_{randint(100, 999)}_manual.jpg"))
        else:
            break


if __name__ == "__main__":
    main()

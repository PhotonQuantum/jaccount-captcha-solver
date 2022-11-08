import re
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from os import path, mkdir
from queue import SimpleQueue
from random import randint
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import httpx
import pytesseract
from PIL import Image

import ocr
import ocr_legacy

# Modify this to change worker count
WORKERS = 9
# Modify this to fetch more/less images
FETCH_COUNT = 1000
# Modify this to change recognize mode
REC_MODE = 0  # 0: tesseract 1: svm 2: resnet
# Select ONNX to test your converted model
MODEL_TYPE = 0  # 0: raw 1: onnx
# Modify this to change working mode
#
# 0: dataset preparing (fetch captcha from jaccount, label them by using existing
# OCR tools, and save both labelled and unlabelled(failed) images.)
#
# 1: benchmark (just test the model and don't save any file.)
#
WORK_MODE = 0

LOGIN_URL = "https://i.sjtu.edu.cn/jaccountlogin"
LOGIN_POST_URL = "https://jaccount.sjtu.edu.cn/jaccount/ulogin"
CAPTCHA_URL = "https://jaccount.sjtu.edu.cn/jaccount/captcha"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:71.0) Gecko/20100101 Firefox/71.0",
           "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8"}

DATA_DIR = path.join(path.dirname(path.abspath(__file__)), 'labelled')
FAIL_DIR = path.join(path.dirname(path.abspath(__file__)), 'unlabelled')
if not path.exists(DATA_DIR):
    mkdir(DATA_DIR)
if not path.exists(FAIL_DIR):
    mkdir(FAIL_DIR)


def recognize_captcha(captcha_img):
    """ tesseract-based recognizer """
    raw_png = BytesIO(captcha_img)
    img = Image.open(raw_png)
    img = img.convert("L")
    table = [0] * 156 + [1] * 100
    img = img.point(table, '1')
    result = pytesseract.image_to_string(img, config="-c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyz --psm 7")
    return result.strip()

def login(client, username, password, recognizer=None):
    ''' a slightly modified login function from pysjtu '''
    login_page_req = client.get(LOGIN_URL)
    uuid = re.findall(r"(?<=uuid\": ').*(?=')", login_page_req.text)[0]
    login_params = parse_qs(urlparse(str(login_page_req.url)).query)
    login_params = {k: v[0] for k, v in login_params.items()}

    captcha_img = client.get(CAPTCHA_URL,
                             params={"uuid": uuid, "t": int(time.time() * 1000)}).content

    if recognizer:
        captcha = recognizer.recognize(captcha_img)
    else:
        captcha = recognize_captcha(captcha_img)

    login_params.update({"v": "", "uuid": uuid, "user": username, "pass": password, "captcha": captcha})
    result = client.post(LOGIN_POST_URL, params=login_params, headers=HEADERS)
    if "Wrong username or password" in result.text:
        return captcha, captcha_img
    else:
        return None, captcha_img


def fetch_thread(queue: SimpleQueue, rtn_queue: SimpleQueue):
    """ main worker """
    client = httpx.Client(timeout=httpx.Timeout(3))

    # Load recognizer according to global const.
    if REC_MODE == 0:
        recognizer = None
    elif REC_MODE == 1:
        if MODEL_TYPE == 0:
            recognizer = ocr_legacy.SVMRecognizer()
        else:
            recognizer = ocr.LegacyRecognizer()
    else:
        if MODEL_TYPE == 0:
            recognizer = ocr_legacy.NNRecognizer()
        else:
            recognizer = ocr.NNRecognizer()
    while True:
        # get the task
        i = queue.get()

        # self-kill when queue is empty
        if i == -1:
            queue.put(-1)
            print("Worker quit.")
            break

        # login and get the captcha
        try:
            rtn = login(client, str(randint(1000, 9999)), str(randint(1000, 9999)), recognizer)
        except:
            # ignore network errors. (using bare exceptions due to unstable httpx exception API.)
            print(f"{i} - FATAL")
            client.cookies = {}
            continue
        captcha, img = rtn

        if captcha:
            # save the labelled captcha
            if WORK_MODE == 0:
                with open(path.join(DATA_DIR, f"{captcha}_{randint(100, 999)}.jpg"), mode="wb") as f:
                    f.write(img)

            # report the result
            print(f"{i} - SUCC - {captcha}")
            rtn_queue.put(1)
        else:
            # save the failed image
            if WORK_MODE == 0:
                with open(path.join(FAIL_DIR, f"{uuid4().hex}.jpg"), mode="wb") as f:
                    f.write(img)

            # report the result
            print(f"{i} - FAIL")
            rtn_queue.put(0)

        # purge cookies to simulate a clean session next time.
        client.cookies = {}


def main():
    # setup worker and result queues
    queue = SimpleQueue()
    rtn_queue = SimpleQueue()
    for i in range(FETCH_COUNT):
        queue.put(i)
    queue.put(-1)

    # bring up workers
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        for i in range(WORKERS):
            executor.submit(fetch_thread, queue, rtn_queue)
            print(f"Worker {i} online.")
    print("Finished.")

    # collect results
    rtn_queue.put(-1)
    rtn = rtn_queue.get()
    succ = 0
    fail = 0
    while rtn != -1:
        if rtn == 1:
            succ += 1
        else:
            fail += 1
        rtn = rtn_queue.get()

    # final report
    print(f"[SUCC] {succ} [FAIL] {fail}")


if __name__ == "__main__":
    main()

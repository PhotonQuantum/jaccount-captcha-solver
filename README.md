# Jaccount Captcha Solver

High accuracy captcha solver for SJTU Jaccount login page using SVM and ResNet.

![captcha example](screenshots/captcha.jpg)
[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2FPhotonQuantum%2Fjaccount-captcha-solver.svg?type=shield)](https://app.fossa.io/projects/git%2Bgithub.com%2FPhotonQuantum%2Fjaccount-captcha-solver?ref=badge_shield)

This work includes two high performance recognizers.

The SVM based recognizer has an accuracy of 90%. It first applies projection-based algorithm to the input image, then use a pre-trained SVM model
to predict the answer. It's memory and cpu efficient.

The Resnet based recognizer achieves an average accuracy of 99%. It feeds the image directly into a pre-trained ResNet-20 model to predict the answer.
It consumes more memory and computing power.

## Getting Started

These instructions will get you a copy of the project up for development and testing purposes.

### Prerequisites

- Python >= 3.7
- OpenCV >= 2.0
- Tesseract

Optionally,

- A CUDA device

### Installing

First, clone this repository.

```shell script
$ git clone https://github.com/PhotonQuantum/jaccount-captcha-solver
```

It's strongly recommended to setup a virtual environment first to avoid polluting global packages.

```shell script
$ python -m venv .venv
$ source ./.venv/bin/activate
```

Then you may install required packages.

```shell script
$ pip install -r requirements.txt
```

Be aware that `torchvision == 0.2.2` has a known compatibility issue with `Pillow >= 7.0.0`. If you met with problems when importing `torchvision`, you may work it around by applying the given patch.

```shell script
$ patch -p0 -d .venv/lib/python* < torchvision.patch
```

Now you are ready to go!

*For instructions on fetching a dataset, training your model, and benchmarking it, please refer to [USAGE.md](USAGE.md) for more information.*

## Deployment

### pysjtu

You need to compile your model into ONNX format if you haven't done so yet. For instructions, see [USAGE.md](usage.md).

NNRecognizer feeds the whole captcha image into your model.
Any model with an input of `[1, 1, *, *]` and an output of `[tensor[26], tensor[26], tensor[26], tensor[26], tensor[27]]` is accepted by `NNRecognizer`.

LegacyRecognizer feeds segmented images (each contains only one character) into your model. Any model with an input of `[*, *]` and an output of `[str, ...]` is accepted by `LegacyRecognizer`.

```python
from pysjtu import LegacyRecognizer, NNRecognizer, Session
svm_ocr = LegacyRecognizer(model_file="svm_model.onnx")
nn_ocr = NNRecognizer(model_file="nn_model.onnx")

session = Session(ocr=svm_ocr)  # or Session(ocr=nn_ocr)
```

### Other

`ocr_legacy.py` provides `SVMRecognizer` and `NNRecognizer`. Either of them accepts raw models. It depends on `nn_models.py` and `utils.py`.

`ocr.py` provides `LegacyRecognizer` and `NNRecognizer`. Either of them accepts ONNX models. It depends on `utils.py`.

You may integrate them into your project:

```python
>>> import PIL from Image
>>> from ocr_legacy import NNRecognizer
>>> recognizer = NNRecognizer(model_file="ckpt.pth")
>>> img = Image.open("captcha.jpg")
>>> recognizer.recognize(img)
gbmke
```

## Built With

* [PyTorch](https://pytorch.org/) - An open source machine learning framework that accelerates the path from research prototyping to production deployment.
* [pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10) - a proper ResNet implementation for CIFAR10/CIFAR100 in pytorch.
* [SciKit-Learn](https://scikit-learn.org/) - A free software machine learning library for the Python programming language.
* [Tesseract](https://github.com/tesseract-ocr/tesseract/) - an OCR engine with support for unicode and the ability to recognize more than 100 languages out of the box.
* [Pillow](https://python-pillow.org/) - The friendly PIL fork.
* [Matplotlib](https://matplotlib.org/) - A Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms.
* [HTTPX](https://www.python-httpx.org/) - A next generation HTTP client for Python.
* [OpenCV](https://opencv.org/) - An open source computer vision and machine learning software library.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details


[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2FPhotonQuantum%2Fjaccount-captcha-solver.svg?type=large)](https://app.fossa.io/projects/git%2Bgithub.com%2FPhotonQuantum%2Fjaccount-captcha-solver?ref=badge_large)

## Acknowledgments

- [T.T. Tang](https://github.com/EletronicElephant) for his idea and support on training a ResNet-20 model 
to do end-to-end multi-task learning on captcha images.

- [Yerlan Idelbayev](https://github.com/akamaster) for his ResNet implementation.
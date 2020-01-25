# a slightly modified ocr.py from pysjtu
import pickle
from io import BytesIO

from utils import *


class Recognizer:
    """ Base class for Recognizers """
    pass


class SVMRecognizer(Recognizer):
    """
    An SVM-based captcha recognizer.

    It first applies projection-based algorithm to the input image, then use a pre-trained SVM model
    to predict the answer.

    It's memory and cpu efficient. The accuracy is around 90%.
    """

    def __init__(self, model_file: str = "model.pickle"):
        self._classifier = pickle.load(open(model_file, mode="rb"))
        self._table = [0] * 156 + [1] * 100

    def recognize(self, img: bytes):
        """
        Predict the captcha.

        :param img: An PIL Image containing the captcha.
        :return: captcha in plain text.
        """
        img_rec = Image.open(BytesIO(img))
        img_rec = img_rec.convert("L")
        img_rec = img_rec.point(self._table, "1")

        segments = [normalize(v_split(segment)).convert("L").getdata() for segment in h_split(img_rec)]
        return "".join(self._classifier.predict(segments))


class NNRecognizer(Recognizer):
    """
    A ResNet-20 based captcha recognizer.

    It feeds the image directly into a pre-trained ResNet-20 model to predict the answer.

    It consumes more memory and computing power than :class:`SVMRecognizer`. The accuracy is around 98%.

    This recognizer requires pytorch and torchvision to work.

    .. note::

        You may set the flag `use_cuda` to speed up predicting, but be aware that it takes time to load the model
        into your GPU and there won't be significant speed-up unless you have a weak CPU.
    """

    def __init__(self, model_file: str = "ckpt.pth", use_cuda=False):
        import torch
        from torchvision import transforms
        from nn_models import resnet20
        self._table = [0] * 156 + [1] * 100
        self._use_cuda = use_cuda
        if self._use_cuda:
            self._model = resnet20().cuda()
            checkpoint = torch.load(model_file)
        else:
            self._model = resnet20().cpu()
            cpu_device = torch.device("cpu")
            checkpoint = torch.load(model_file, map_location=cpu_device)
        self._model.load_state_dict(checkpoint["net"])
        self._model.eval()
        self._loader = transforms.ToTensor()

    @staticmethod
    def tensor_to_captcha(tensors):
        """
        A helper function to translate Tensor prediction to str.

        :param tensors: prediction in Tensor.
        :return: prediction in str.
        """
        rtn = ""
        for tensor in tensors:
            if int(tensor) != 26:
                rtn += chr(ord("a") + int(tensor))

        return rtn

    def recognize(self, img: bytes):
        """
        Predict the captcha.

        :param img: An PIL Image containing the captcha.
        :return: captcha in plain text.
        """
        from torch.autograd import Variable
        img_rec = Image.open(BytesIO(img))
        img_rec = img_rec.convert("L")
        img_rec = img_rec.point(self._table, "1")
        img_tensor = self._loader(img_rec).float().unsqueeze(0)
        if self._use_cuda:
            img_tensor = Variable(img_tensor).cuda()
        else:
            img_tensor = Variable(img_tensor).cpu()

        output = self._model(img_tensor)
        predicted_tensor = [tensor.max(1)[1] for tensor in output]
        predicted = NNRecognizer.tensor_to_captcha(predicted_tensor)
        return predicted

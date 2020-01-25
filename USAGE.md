# Prepare the dataset

## Use pre-fetched dataset

You may get pre-fetched dataset from releases page. Download and decompress it, and you will get folders named `labelled` and `segmented`.

## Fetch dataset on your own

You may crawl jaccount login page using `crawler.py` to get captcha images. Pre-trained models can be used to help you label them.

Don't forget to change related settings in `crawler.py`. You should set `REC_MODE` to the mode you want and set `WORK_MODE` to 0.

```python
# Modify this to change recognize mode
REC_MODE = 0  # 0: tesseract 1: svm 2: resnet
# Modify this to change working mode
#
# 0: dataset preparing (fetch captcha from jaccount, label them by using existing
# OCR tools, and save both labelled and unlabelled(failed) images.)
#
# 1: benchmark (just test the model and don't save any file.)
#
WORK_MODE = 0
```

> Note: You need to have tesseract installed before using tesseract engine.
> 
> `model.pickle` or `ckpt.pth` should be put in the same directory if you are going to use SVM or ResNet-10 engine respectively.

Then you may get your dataset ready.

```shell script
$ python crawler.py
Worker 0 online.
Worker 1 online.
...
Worker 8 online.
1 - SUCC - ayqx
2 - SUCC - fjxie
4 - FAIL
3 - SUCC - dyxe
6 - FATAL
...
Worker quit.
999 - SUCC - fjsl
Worker quit.
Finished.
[SUCC] 874 [FAIL] 118
```

Now you will have 874 images in `labelled` dir and 118 images in `unlabelled` dir.

# Train your model

## SVM

### Dataset

To train the SVM model, a dataset contains 4000+ images is ideal.

### Segmentation

First, you should segment all captcha.

```shell script
$ python segment.py
```

This script will run for about 5-30 secs, and you will get your labelled images segmented under `segmented` dir.

### Training

Remember to backup the previous model `model.pickle`, then you may train your own SVM model.

```shell script
$ python svm_train.py
              precision    recall  f1-score   support

           a       1.00      1.00      1.00        81
           b       1.00      1.00      1.00        75
           c       0.98      1.00      0.99        81
...
           y       1.00      1.00      1.00        89
           z       1.00      1.00      1.00        94

    accuracy                           1.00      2157
   macro avg       1.00      1.00      1.00      2157
weighted avg       1.00      1.00      1.00      2157

Confusion matrix:
...
```

This script will run for about 10-60 secs, then you will get your new model `model.pickle`.

## ResNet-20

### Dataset

To train the ResNet-20 model, a dataset contains at least 20000 images is ideal. Smaller dataset may cause the model to overfit, or fail to converge.

### Adjust settings

Before running the script, make sure you read `nn_train.py` and change settings according to your environment and usage. Using default settings may result in fatal errors or broken models.

For example, you may change `USE_CUDA` to False if you don't have a CUDA device. And you may lower `batch_size` if your GPU has a smaller graphics memory.

### Training

Remember to backup the previous model `ckpt.pth`.

The ResNet-20 model doesn't need any segmentation, so just run:

```shell script
$ python nn_train.py
==> Preparing data..
==> Building model..
Epoch [0] Batch [1/82] Loss: 21.787 | Traininig Acc: [Sentence] 0.000% (0/320) [Char] 3.375% (54/1600)
Epoch [0] Batch [2/82] Loss: 20.357 | Traininig Acc: [Sentence] 0.000% (0/640) [Char] 3.312% (106/3200)
...
Epoch [0] Batch [82/82] Loss: 14.777 | Traininig Acc: [Sentence] 0.000% (0/25989) [Char] 16.881% (21936/129945)
==> Testing...
Test Acc: [Sentence] 0.000000 [Char] 19.101124
Saving..
train_loss: 14.777308417529595
Epoch [1] Batch [1/82] Loss: 13.371 | Traininig Acc: [Sentence] 0.000% (0/320) [Char] 21.500% (344/1600)
Epoch [1] Batch [2/82] Loss: 13.433 | Traininig Acc: [Sentence] 0.000% (0/640) [Char] 20.656% (661/3200)
...
...
Epoch [50] Batch [82/82] Loss: 0.001 | Traininig Acc: [Sentence] 100.000% (25989/25989) [Char] 100.000% (129945/129945)
==> Testing...
Test Acc: [Sentence] 99.399723 [Char] 99.873788
Saving..
train_loss: 0.001274651560290694
```

According to your compute power, dataset size and hyper parameters, 
the training time may vary from several minutes to days.
For example, a dataset of 32486 images takes a laptop with NVIDIA GTX 1650 an hour to train with epoch=50 and 
batch_size=320.

If the script finishes, the latest checkpoint will be copied to `ckpt.pth` in the current directory.

Sometimes you may want to terminate the script in advance (for the current accuracy is satisfying), 
you may get the trainned model from `checkpoint` directory. 
The file name pattern is `ckpt_<epoch>_acc_<test_accuracy>.pth`

# Benchmarking

You may want to benchmark your trained model using `crawler.py`.

Don't forget to adjust the `WORK_MODE` flag to 1 and `REC_MODE` to the model you want to benchmark on.

For example:

```python
REC_MODE = 2
WORK_MODE = 1
```

```shell script
$ python crawler.py
Worker 0 online.
Worker 1 online.
...
Worker 8 online.
7 - SUCC - kjzh
6 - SUCC - cqmjw
4 - SUCC - xrgym
...
997 - SUCC - nompq
Worker quit.
Finished.
[SUCC] 988 [FAIL] 8
```

And you may calculate the accuracy. In this case: 

```shell script
; 988/(988+8)
	~0.99196787148594377510
```

# Troubleshooting

## Overfit

It's common for models using Adam optimizer to slightly overfit, 
so don't be panic if you see difference between train accuracy and test accuracy. 

For example, if you see something like:

```
Epoch [5] Batch [81/82] Loss: 0.461 | Traininig Acc: [Sentence] 91.046% (23599/25920) [Char] 98.057% (127082/129600)
Epoch [5] Batch [82/82] Loss: 0.461 | Traininig Acc: [Sentence] 91.054% (23664/25989) [Char] 98.058% (127422/129945)
==> Testing...
Test Acc: [Sentence] 46.513776 [Char] 85.870402
```

Or your test accuracy drops from 90%+ to 70%+.

Be relax. It will be corrected by the network or lr_scheduler.

## Serious overfit

However, if you see test accuracy drop dramatically, for example:

```
Epoch [22] Batch [82/82] Loss: 0.027 | Traininig Acc: [Sentence] 99.519% (25864/25989) [Char] 99.902% (129818/129945)
==> Testing...
Test Acc: [Sentence] 97.706634 [Char] 99.529013
...
Epoch [23] Batch [82/82] Loss: 0.043 | Traininig Acc: [Sentence] 99.084% (25751/25989) [Char] 99.815% (129705/129945)
==> Testing...
Test Acc: [Sentence] 0.000000 [Char] 13.667847
```

Unfortunately, there may be a serious overfitting.

But don't be panic, you may just wait for lr_scheduler to kick in to correct it. Sometimes the loss increasement will trigger a lr decay 
and solve the problem. Your test accuracy will increase gradually.

If lr_scheduler can't correct this, then you may take the following measures:

- If the accuracy is satisfying in a previous epoch, take that checkpoint as the accepted model.
- If your dataset is smaller than 10000 images, you should crawl more images. Such a small dataset will cause the model
to overfit easily.
- You may decrease the initial lr by editing `nn_train.py`.
- You may consider replacing Adam optimizer with SGD optimizer, for SGD optimizer usually has better generalization 
performance than Adam optimizer. Don't forget to adjust lr accordingly (SGD usually needs a higher lr to converge faster).

### Model doesn't converge

It's rare for such a small model on this dataset to fail. If it fails to converge, please make sure your dataset is 
reasonable. Check for any label error and make sure images look right. Crawling more images may also solve the problem.


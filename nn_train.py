import os
from os import listdir, path
from shutil import copyfile

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, random_split, DataLoader

from nn_models import resnet20

# Train config
USE_CUDA = False
USE_MPS = True  # for Apple silicon devices
TEST_FACTOR = 0.2
# Hyper parameters
start_epoch = 0
end_epoch = 75
lr = 0.01
batch_size = 640  # You may adjust this according to your graphics memory. Don't forget to adjust lr.


# Also check CHECKPOINT SETTINGS below.


class CaptchaSet(Dataset):
    def __init__(self, root, transform):
        self._table = [0] * 156 + [1] * 100
        self.transform = transform

        self.root = root
        self.imgs = listdir(root)

    @staticmethod
    def _get_label_from_fn(fn):
        raw_label = fn.split("_")[0]
        labels = [ord(char) - ord("a") for char in raw_label]
        if len(labels) == 4: labels.append(26)
        return labels

    def __getitem__(self, idx):
        img = Image.open(path.join(self.root, self.imgs[idx])).convert("L")
        img = img.point(self._table, "1")

        label = CaptchaSet._get_label_from_fn(self.imgs[idx])

        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)


mps_device = None
if USE_MPS:
    mps_device = torch.device("mps")


def transfer_to_device(x):
    if USE_CUDA:
        return x.cuda()
    elif mps_device is not None:
        return x.to(mps_device)
    else:
        return x


# Data pre-processing
print('==> Preparing data..')
transform = transforms.ToTensor()

dataset = CaptchaSet(root="labelled", transform=transform)
test_count = int(len(dataset) * TEST_FACTOR)
train_count = len(dataset) - test_count
train_set, test_set = random_split(dataset, [train_count, test_count])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

# Model
print('==> Building model..')
model = resnet20()
model = transfer_to_device(model)

optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
criterion = nn.CrossEntropyLoss()
criterion = transfer_to_device(criterion)

# CHECKPOINT SETTINGS
# If you want to restore training (instead of training from beginning),
# you can continue training based on previously-saved models
# WARNING: BEWARE that there may be some problems with this implementation currently. You may get inconsistent results.
restore_model_path = None
# restore_model_path = 'checkpoint/ckpt_0_acc_0.000000.pth'
if restore_model_path:
    checkpoint = torch.load(restore_model_path)
    model.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch'] + 1
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])


def tensor_to_captcha(tensors):
    rtn = ""
    for tensor in tensors:
        if int(tensor) == 26:
            rtn += " "
        else:
            rtn += chr(ord("a") + int(tensor))

    return rtn


def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    per_char_correct = 0
    per_char_total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = transfer_to_device(inputs)

        # Learn and predict
        optimizer.zero_grad()
        outputs = model(inputs)

        targets = [transfer_to_device(target) for target in targets]

        # Calculate loss
        losses = []
        for idx in range(len(outputs)):
            losses.append(criterion(outputs[idx], targets[idx]))
        loss = sum(losses)
        loss.backward()
        train_loss += loss.item()

        # Optimizer step
        optimizer.step()

        # Calculate accuracy (sentence based and char based)
        predicted = torch.stack([tensor.max(1)[1] for tensor in outputs], 1)
        targets_stacked = torch.stack(targets, 1)
        per_char_total += targets[0].size(0) * 5
        per_char_correct += predicted.eq(targets_stacked).sum().item()
        total += targets[0].size(0)
        correct += torch.all(predicted.eq(targets_stacked), 1).sum().item()
        batch_idx_last = batch_idx

        # Report statistics
        print('Epoch [%d] Batch [%d/%d] Loss: %.3f | Traininig Acc: [Sentence] %.3f%% (%d/%d) [Char] %.3f%% (%d/%d)'
              % (epoch, batch_idx + 1, len(train_loader), train_loss / (batch_idx + 1),
                 100. * correct / total, correct, total, 100. * per_char_correct / per_char_total, per_char_correct,
                 per_char_total))

    # Return train_loss for lr_scheduler
    return train_loss / (batch_idx_last + 1)


def test(epoch):
    print('==> Testing...')
    model.eval()
    total = 0
    correct = 0
    per_char_correct = 0
    per_char_total = 0
    with torch.no_grad():
        ##### TODO: calc the test accuracy #####
        # Hint: You do not have to update model parameters.
        #       Just get the outputs and count the correct predictions.
        #       You can turn to `train` function for help.
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = transfer_to_device(inputs)
            targets = [transfer_to_device(target) for target in targets]

            # Predict
            outputs = model(inputs)

            # Calculate accuracy (sentence-based and char-based)
            predicted = torch.stack([tensor.max(1)[1] for tensor in outputs], 1)
            targets_stacked = torch.stack(targets, 1)
            total += targets[0].size(0)
            correct += torch.all(predicted.eq(targets_stacked), 1).sum().item()
            per_char_total += targets[0].size(0) * 5
            per_char_correct += predicted.eq(targets_stacked).sum().item()
        acc = 100. * correct / total
        per_char_acc = 100. * per_char_correct / per_char_total
        ########################################
    # Save checkpoint.
    print('Test Acc: [Sentence] %f [Char] %f' % (acc, per_char_acc))
    print('Saving..')
    state = {
        'net': model.state_dict(),
        'scheduler': scheduler.state_dict(),
        'optimizer': optimizer.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt_%d_acc_%f.pth' % (epoch, acc))

    return './checkpoint/ckpt_%d_acc_%f.pth' % (epoch, acc)


for epoch in range(start_epoch, end_epoch + 1):
    train_loss = train(epoch)
    ckpt_file = test(epoch)
    print(f"train_loss: {train_loss}")

    # Scheduler step
    scheduler.step(train_loss)

copyfile(ckpt_file, "./ckpt.pth")

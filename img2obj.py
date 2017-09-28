from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.image as img
from PIL import Image
import numpy as np
# import cv2
import time

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR100 HW05 BME 595')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR100('../data', train=True, download=False,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize(mean, std)
                      ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR100('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=5)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 100)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # print(len(x))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # print(x)
        x = x.view(-1, 100)
        # print(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.softmax(x)

    def train(self):
        # model.train()
        for epoch in range(0, args.epochs):
            for batch_idx, (image, target) in enumerate(train_loader):
                if args.cuda:
                    image, target = image.cuda(), target.cuda()
                image, target = Variable(image), Variable(target)
                # print("Image", image.data)
                optimizer.zero_grad()
                # Can also be written as: output = model.(image)
                output = model.forward(image)
                # print(output.data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(image), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.data[0]))

    def test(self):
        test_loss = 0
        correct = 0
        for image, target in test_loader:
            if args.cuda:
                image, target = image.cuda(), target.cuda()
            image, target = Variable(image, volatile=True), Variable(target)
            output = model(image)
            test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr)

# model.train()
# torch.save(model.state_dict(), 'save_model.pt')

model.load_state_dict(torch.load('save_model.pt'))

# model.test()


# # Camera part
# # Camera 0 is the integrated web cam on my netbook
# camera_port = 0

# # Now we can initialize the camera capture object with the cv2.VideoCapture class.
# # All it needs is the index to a camera port.
# camera = cv2.VideoCapture(camera_port)

# Captures a single image from the camera and returns it in PIL format


# def get_image():
#     # read is the easiest way to get a full image out of a VideoCapture object.
#     retval, im = camera.read()
#     return im


# print("Taking images...")

# t_end = time.time() + 60 * 0.2

# while time.time() < t_end:
#     camera_capture = get_image()
#     resized_image = camera_capture.resize((32, 32))
#     cv2.imshow('frame', camera_capture)
#     input_image = resized_image
#     # input_image = torch.from_numpy(input_image)
#     input_image = Variable(input_image)
#     # input_image = input_image.view(1, 3, 32, 32)
#     pred = model.forward(input_image)
#     pred = pred.data.numpy()
#     pred_label = np.argmax(pred, axis=1)
#     print(pred_label)


# You'll want to release the camera, otherwise you won't be able to create a new
# capture object until your script exits
# del(camera)
# print("Camera deleted...")

input_image = img.imread("automobile10.png")
input_image = torch.from_numpy(input_image)
input_image = Variable(input_image)
input_image = input_image.view(1, 3, 32, 32)
pred = model.forward(input_image)
pred = pred.data.numpy()
pred_label = np.argmax(pred, axis=1)
print(pred_label)

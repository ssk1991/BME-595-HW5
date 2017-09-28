# BME-595-HW5
## The LeNet 5 architecture was used to develop a classifier for the MNIST and CIFAR100 image sets. 
## The architecture involves 2 Convolution, 2 Max-Pooling, and 2 Fully-Connected layers. Although not originally in the LeNet 5, I incorporated Dropout and a Softmax output layer. The ReLU activation function was used to impart non-linarity.

# Training on the MNIST image set resulted in a testing accuracy of 98.4% after 30 Epochs.
## I believe there is some amount of overfitting which can be avoided by using data augmentations (random crop, flips, rotations, etc.)
![alt text](https://github.com/ssk1991/BME-595-HW5/blob/master/Images/Lenet5%20CIFAR100.PNG)


# Training on the CIFAR100 image set resulted in a testing accuracy of 28.3% after 60 Epochs.
## Once again I think the network tends to overfit the data, rather than learn the features in the images. Data augmentations may be a good alternative to avoid this overfitting. 
![alt text](https://github.com/ssk1991/BME-595-HW5/blob/master/Images/lenet%20MNIST.PNG)


## Possibly using the ELU activation function might be benficial as well.

## The LeNet5 trained on CIFAR100, is able to accept and classify colour input images of size [32x32]. However, I could not execute the program on images captured using the webcam through OpenCV. This is because I was using a Linux server to execute the program, and could not figure how to link the webcam to the server.

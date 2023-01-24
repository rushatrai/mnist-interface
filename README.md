# mnist-interface

A Pytorch implementation of a customized LeNet-5 algorithm desgined to give best results on the classic MNIST dataset.

Colab Notebook walkthrough available at: https://colab.research.google.com/drive/1McPsydqm83FgbTtj-1Az4ZP4Am1hJEeQ#scrollTo=K51iaSqWYpwz

Live version available at: https://huggingface.co/spaces/equ1/mnist_interface

Please click "Restart Space" if it has timed out due to inactivity.

# Implementation details
The `model.py` file contains a PyTorch implementation of a modified LeNet-5 architecture built to perform well on MNIST.

### Original LeNet-5 architecture

ConvNet --> Pool --> ConvNet --> Pool --> (Flatten) --> FullyConnected --> FullyConnected --> Softmax

Results in 30 epochs:

**loss** - 0.0016

**train_accuracy** - 0.9998

**val_loss** - 0.0412

**val_accuracy** - 0.9905

![architecture](https://github.com/guptajay/Kaggle-Digit-Recognizer/raw/master/img/LeNet5.png)

### [Modified LeNet architecture](https://github.com/guptajay/Kaggle-Digit-Recognizer):

ConvNet --> ConvNet --> BatchNorm --> Pool --> (Dropout) --> ConvNet --> ConvNet --> BatchNorm --> Pool --> (Dropout) --> (Flatten) --> FullyConnected --> BatchNorm --> FullyConnected --> BatchNorm --> FullyConnected --> BatchNorm --> (Dropout) --> Softmax

This architecture suggested in the linked blog is designed to further reduce bias and variance by adding augmentations, learning rate decay, BatchNorm and Dropout to achieve an improvement over the already impressive 99.05% accuracy of the original LeNet-5.

# Results
The model is able to achieve a 99.59% validation accuracy within 30 epochs.

Note: It is documented that the performance plateaus at 35 epochs so futher training will likely not improve performance. Considering the simplicity of the dataset, I also doubt if further augmentations or hyperparameter optimization will lead to more than negligible improvements. With this in mind, it is safe to say that this is near the best performance achievable on this dataset without ridiculous overfitting - we have something to be proud of here!


# References
[LeCun et al., Gradient-based learning applied to document recognition (1998)](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)

[Jay Gupta's GitHub repository](https://github.com/guptajay/Kaggle-Digit-Recognizer)



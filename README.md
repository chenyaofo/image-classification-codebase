# Image Classification Codebase

This project **aims** to provide a codebase for the image classification task implemented by PyTorch.
It does not use any high-level deep learning libraries (such as pytorch-lightening or MMClassification).
Thus, it should be easy to follow and modified.

## Requirements

The code is tested on `python==3.9, pyhocon==0.3.57, torch=1.8.0, torchvision=0.9.0`

## Get Started

You can get started with a resnet20 convolution network on cifar10 with the following command.

**Single node, single GPU:**

```bash
deepspeed --include localhost:1 entry/run.py --conf conf/vit_cifar10.conf -o test/0
```

You can use multiple GPUs to accelerate the training with distributed data parallel:

**Single node, multiple GPUs:**

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m entry.run --world-size 2 \
--conf conf/cifar10.conf -o output/cifar10/resnet20
```

**Multiple nodes:**

Node 0:
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m entry.run --world-size 4 --dist-url \
'tcp://IP_OF_NODE0:FREEPORT' --node-rank 0 --conf conf/cifar10.conf -o output/cifar10/resnet20
```

Node 1:
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m entry.run --world-size 4 --dist-url \
'tcp://IP_OF_NODE1:FREEPORT' --node-rank 1 --conf conf/cifar10.conf -o output/cifar10/resnet20
```


## Features

This codebase adopt configuration file (`.hocon`) to store the hyperparameters (such as the learning rate, training epochs and etc.).
If you want to modify the configuration hyperparameters, you have two ways:

1. Modify the configuration file to generate a new file.

2. You can add `-M` in the running command line to modify the hyperparameters temporarily.


For example, if you hope to modify the total training epochs to 100 and the learning rate to 0.05. You can run the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python -m entry.run --conf conf/cifar10.conf -o output/cifar10/resnet20 -M 'max_epochs=100' 'optimizer.lr=0.05'
```

If you modify a non existing hyperparameter, the code will raise an exception.

To list all valid hyperparameters names, you can run the following command:

```bash
pyhocon -i conf/cifar10.conf -f properties
```

## Cite

```
@misc{chen2020image,
  author = {Yaofo Chen},
  title = {Image Classification Codebase},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/chenyaofo/image-classification-codebase}}
}
```

## License

MIT License
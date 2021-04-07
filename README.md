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
CUDA_VISIBLE_DEVICES=0 python -m entry.run --conf conf/cifar.conf -o output/cifar_resnet20
```

You can use multiple GPUs to accelerate the training with distributed data parallel:

**Single node, multiple GPUs:**

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m entry.run --world-size 2 \
--conf conf/cifar.conf -o output/cifar_resnet20
```

**Multiple nodes:**

Node 0:
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m entry.run --world-size 4 --dist-url \
'tcp://IP_OF_NODE0:FREEPORT' --node-rank 0 --conf conf/cifar.conf -o output/cifar_resnet20
```

Node 1:
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m entry.run --world-size 4 --dist-url \
'tcp://IP_OF_NODE0:FREEPORT' --node-rank 1 --conf conf/cifar.conf -o output/cifar_resnet20
```


## Highlights

 - Distributed training support (Use native Pytorch API).
 - DALI data processing support.

## Roadmap
  
  - Prefetch Dataloader
  - AMP support.
  - Auto recomputation support.
  - Auto resume from checkpoint.
  - AutoAugment and RandAugment support.
  - ZeRO-Offload sopport?
  - Other tricks.

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
## Throughput Benchmark
We test this code on NVIDIA A100 and report the throughput in the followings.

| settings | throughput (imgs/s) |
| --- | --- |
| baseline | 928 |
| +channels_last | 992 |
| +amp | 1459 |

> Check for NVIDIA impl and **Throughput Benchmark** at https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/ConvNets/resnet50v1.5/README.md#training-performance-results

## Test Environment

```
PyTorch version: 1.12.1+cu113
Is debug build: False
CUDA used to build PyTorch: 11.3
ROCM used to build PyTorch: N/A

OS: Ubuntu 20.04.4 LTS (x86_64)
GCC version: Could not collect
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.31

Python version: 3.10.6 | packaged by conda-forge | (main, Aug 22 2022, 20:35:26) [GCC 10.4.0] (64-bit runtime)
Python platform: Linux-4.15.0-192-generic-x86_64-with-glibc2.31
Is CUDA available: True
CUDA runtime version: Could not collect
GPU models and configuration: GPU 0: NVIDIA A100-SXM4-40GB
Nvidia driver version: 470.129.06
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

Versions of relevant libraries:
[pip3] numpy==1.23.3
[pip3] pytorch-lightning==1.7.6
[pip3] torch==1.12.1+cu113
[pip3] torchaudio==0.12.1+cu113
[pip3] torchmetrics==0.9.3
[pip3] torchvision==0.13.1+cu113
[conda] numpy                     1.23.3          py310h53a5b5f_0    conda-forge
[conda] pytorch-lightning         1.7.6                    pypi_0    pypi
[conda] torch                     1.12.1+cu113             pypi_0    pypi
[conda] torchaudio                0.12.1+cu113             pypi_0    pypi
[conda] torchmetrics              0.9.3                    pypi_0    pypi
[conda] torchvision               0.13.1+cu113             pypi_0    pypi
```
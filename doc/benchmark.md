## Throughput Benchmark

We test this code on NVIDIA A100 and report the throughput in the followings.

| settings | throughput (imgs/s) |
| --- | --- |
| baseline | 928 |
| +channels_last | 992 |
| +amp | 1459 |
| +channels_last&&amp | 2260 |

> Check for NVIDIA impl and **Throughput Benchmark** at https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/ConvNets/resnet50v1.5/README.md#training-performance-results

Test environment:

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

## More Benchmarks on PyTorch 2.0

We test this code on NVIDIA V100 and report the throughput in the followings.

 - Benchmarks on ResNet-50

| settings | throughput (imgs/s) |
| --- | --- |
| baseline | 345 |
| +channels_last | 345 |
| +amp | 774 |
| +channels_last&&amp | 1175 |
| +channels_last&&amp&&compile(default) | 1228 |
| +channels_last&&amp&&compile(default+fullgraph) | 1228 |
| +channels_last&&amp&&compile(reduce-overhead) | 1234 |
| +channels_last&&amp&&compile(max-autotune) | FAIL |

 - Benchmarks on MobileNetV2

| settings | throughput (imgs/s) |
| --- | --- |
| baseline | 813 |
| +channels_last | 420 |
| +amp | 1315 |
| +channels_last&&amp | 2100 |
| +channels_last&&amp&&compile(default) | 2316 |

 - Benchmarks on ShuffleNetV2

| settings | throughput (imgs/s) |
| --- | --- |
| baseline | 2342 |
| +channels_last | 1854 |
| +amp | 3250 |
| +channels_last&&amp | 3862 |
| +channels_last&&amp&&compile(default) | 4711 |

 - Benchmarks on ViT-B16

| settings | throughput (imgs/s) |
| --- | --- |
| baseline | 102 |
| +amp | 360 |
| +amp&&compile(default) | 289 |

 - Benchmarks on SwinTransformer-tiny

| settings | throughput (imgs/s) |
| --- | --- |
| baseline | 264 |
| +amp | 499 |
| +amp&&compile(default) | 789 |

Test environment:

```
PyTorch version: 2.0.0+cu118
Is debug build: False
CUDA used to build PyTorch: 11.8
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.1 LTS (x86_64)
GCC version: (Ubuntu 11.3.0-1ubuntu1~22.04) 11.3.0
Clang version: Could not collect
CMake version: version 3.25.0
Libc version: glibc-2.35

Python version: 3.10.9 | packaged by conda-forge | (main, Feb  2 2023, 20:20:04) [GCC 11.3.0] (64-bit runtime)
Python platform: Linux-5.4.0-139-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 11.8.89
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: Tesla V100-SXM2-32GB
Nvidia driver version: 525.85.12
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.8.7.0
/usr/lib/x86_64-linux-gnu/libcudnn_adv_infer.so.8.7.0
/usr/lib/x86_64-linux-gnu/libcudnn_adv_train.so.8.7.0
/usr/lib/x86_64-linux-gnu/libcudnn_cnn_infer.so.8.7.0
/usr/lib/x86_64-linux-gnu/libcudnn_cnn_train.so.8.7.0
/usr/lib/x86_64-linux-gnu/libcudnn_ops_infer.so.8.7.0
/usr/lib/x86_64-linux-gnu/libcudnn_ops_train.so.8.7.0
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                    x86_64
CPU op-mode(s):                  32-bit, 64-bit
Address sizes:                   46 bits physical, 48 bits virtual
Byte Order:                      Little Endian
CPU(s):                          10
On-line CPU(s) list:             0-9
Vendor ID:                       GenuineIntel
Model name:                      Intel(R) Xeon(R) Platinum 8255C CPU @ 2.50GHz
CPU family:                      6
Model:                           85
Thread(s) per core:              1
Core(s) per socket:              10
Socket(s):                       1
Stepping:                        5
BogoMIPS:                        4999.99
Flags:                           fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl cpuid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch invpcid_single pti fsgsbase bmi1 hle avx2 smep bmi2 erms invpcid rtm mpx avx512f avx512dq rdseed adx smap clflushopt clwb avx512cd avx512bw avx512vl xsaveopt xsavec xgetbv1 arat avx512_vnni
Hypervisor vendor:               KVM
Virtualization type:             full
L1d cache:                       320 KiB (10 instances)
L1i cache:                       320 KiB (10 instances)
L2 cache:                        40 MiB (10 instances)
L3 cache:                        35.8 MiB (1 instance)
NUMA node(s):                    1
NUMA node0 CPU(s):               0-9
Vulnerability Itlb multihit:     KVM: Vulnerable
Vulnerability L1tf:              Mitigation; PTE Inversion
Vulnerability Mds:               Vulnerable: Clear CPU buffers attempted, no microcode; SMT Host state unknown
Vulnerability Meltdown:          Mitigation; PTI
Vulnerability Mmio stale data:   Vulnerable: Clear CPU buffers attempted, no microcode; SMT Host state unknown
Vulnerability Retbleed:          Vulnerable
Vulnerability Spec store bypass: Vulnerable
Vulnerability Spectre v1:        Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:        Mitigation; Retpolines, STIBP disabled, RSB filling, PBRSB-eIBRS Not affected
Vulnerability Srbds:             Not affected
Vulnerability Tsx async abort:   Vulnerable: Clear CPU buffers attempted, no microcode; SMT Host state unknown

Versions of relevant libraries:
[pip3] numpy==1.23.5
[pip3] torch==2.0.0+cu118
[pip3] torchaudio==2.0.1+cu118
[pip3] torchdata==0.6.0
[pip3] torchvision==0.15.1+cu118
[conda] numpy                     1.23.5                   pypi_0    pypi
[conda] torch                     2.0.0+cu118              pypi_0    pypi
[conda] torchaudio                2.0.1+cu118              pypi_0    pypi
[conda] torchdata                 0.6.0                    pypi_0    pypi
[conda] torchvision               0.15.1+cu118             pypi_0    pypi
```
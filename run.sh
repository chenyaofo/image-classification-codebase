# exp01
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
--nproc_per_node=4 --master_port 12345 train.py \
--batch_size 64 --model moga_a --dali  -j 6 \
--max_epochs 150 --criterion ce --lr 0.1 \
--data /path/to/imagenet/ \
-o mobilenet_family/exp01

# exp02
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
--nproc_per_node=4 --master_port 12346 train.py \
--batch_size 64 --model moga_a --dali  -j 6 \
--max_epochs 250 --criterion ce --lr 0.1 \
--data /path/to/imagenet/ \
-o mobilenet_family/exp02

# exp03
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
--nproc_per_node=4 --master_port 12347 train.py \
--batch_size 64 --model moga_a --dali  -j 6 \
--max_epochs 150 --criterion ce --lr 0.05 \
--data /path/to/imagenet/ \
-o mobilenet_family/exp03

# exp04
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
--nproc_per_node=4 --master_port 12348 train.py \
--batch_size 64 --model moga_a --dali  -j 6 \
--max_epochs 150 --criterion labelsmooth --lr 0.1 \
--data /path/to/imagenet/ \
-o mobilenet_family/exp04

# exp05
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
--nproc_per_node=4 --master_port 12345 train.py \
--batch_size 64 --model moga_a --dali  -j 6 \
--max_epochs 150 --criterion ce --lr 0.1 \
--data /path/to/imagenet/ \
--bn_momentum 0.01 \
-o mobilenet_family/exp05

# exp06
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
--nproc_per_node=4 --master_port 12345 train.py \
--batch_size 64 --model moga_a --dali  -j 6 \
--max_epochs 150 --criterion ce --lr 0.1 \
--data /path/to/imagenet/ \
--dropout 0 \
-o mobilenet_family/exp06

# exp07
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
--nproc_per_node=4 --master_port 12345 train.py \
--batch_size 64 --model moga_a --dali  -j 6 \
--max_epochs 150 --criterion ce --lr 0.1 \
--data /path/to/imagenet/ \
--scheduler warmup_exponential \
-o mobilenet_family/exp07
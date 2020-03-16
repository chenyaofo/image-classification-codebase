
try:
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
    from nvidia.dali.pipeline import Pipeline
except ImportError:
    raise ImportError("Please install DALI refer to "
                      "https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/installation.html.")

from utils.distributed import world_size, local_rank


class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, data_dir, crop, device_id=local_rank(), only_cpu=False):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank(),
                                    num_shards=world_size(), random_shuffle=True)
        # let user decide which pipeline works him bets for RN version he runs
        dali_device = 'cpu' if only_cpu else 'gpu'
        decoder_device = 'cpu' if only_cpu else 'mixed'
        # This padding sets the size of the internal nvJPEG buffers to be able to
        # handle all images from full-sized ImageNet
        # without additional reallocations
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        self.decode = ops.ImageDecoderRandomCrop(device=decoder_device, output_type=types.RGB,
                                                 device_memory_padding=device_memory_padding,
                                                 host_memory_padding=host_memory_padding,
                                                 random_aspect_ratio=[0.8, 1.25],
                                                 random_area=[0.1, 1.0],
                                                 num_attempts=100)
        self.res = ops.Resize(device=dali_device, resize_x=crop, resize_y=crop, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device=dali_device,
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        rng = self.coin()
        self.images, self.labels = self.input(name="Reader")
        images = self.decode(self.images)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.labels]


class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, data_dir, crop, size, device_id=local_rank(), only_cpu=False):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank(),
                                    num_shards=world_size(), random_shuffle=False)
        dali_device = 'cpu' if only_cpu else 'gpu'
        decoder_device = 'cpu' if only_cpu else 'mixed'
        self.decode = ops.ImageDecoder(device=decoder_device, output_type=types.RGB)
        self.res = ops.Resize(device=dali_device, resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device=dali_device,
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

    def define_graph(self):
        self.images, self.labels = self.input(name="Reader")
        images = self.decode(self.images)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]

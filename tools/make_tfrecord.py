'''
This script aims to create tfrecord tar shards with multi-processing.
'''

import os
import random
import datetime
from multiprocessing import Process
from torchvision.datasets.folder import ImageFolder

from codebase.torchutils.serialization import jsonpack

import struct
import tfrecord

def create_index(tfrecord_file: str, index_file: str) -> None:
    """
    refer to https://github.com/vahidk/tfrecord/blob/master/tfrecord/tools/tfrecord2idx.py
    Create index from the tfrecords file.
    Stores starting location (byte) and length (in bytes) of each
    serialized record.
    Params:
    -------
    tfrecord_file: str
        Path to the TFRecord file.
    index_file: str
        Path where to store the index file.
    """
    infile = open(tfrecord_file, "rb")
    outfile = open(index_file, "w")

    while True:
        current = infile.tell()
        try:
            byte_len = infile.read(8)
            if len(byte_len) == 0:
                break
            infile.read(4)
            proto_len = struct.unpack("q", byte_len)[0]
            infile.read(proto_len)
            infile.read(4)
            outfile.write(str(current) + " " + str(infile.tell() - current) + "\n")
        except:
            print("Failed to parse TFRecord.")
            break
    infile.close()
    outfile.close()


def make_wds_shards(pattern, num_shards, num_workers, samples, map_func, **kwargs):
    random.shuffle(samples)
    samples_per_shards = [samples[i::num_shards] for i in range(num_shards)]
    shard_ids = list(range(num_shards))
    processes = [
        Process(
            target=write_partial_samples,
            args=(
                pattern,
                shard_ids[i::num_workers],
                samples_per_shards[i::num_workers],
                map_func,
                kwargs
            )
        )
        for i in range(num_workers)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()


def write_partial_samples(pattern, shard_ids, samples, map_func, kwargs):
    for shard_id, samples in zip(shard_ids, samples):
        write_samples_into_single_shard(pattern, shard_id, samples, map_func, kwargs)


def write_samples_into_single_shard(pattern, shard_id, samples, map_func, kwargs):
    fname = pattern % shard_id
    print(f"[{datetime.datetime.now()}] start to write samples to shard {fname}")
    # stream = TarWriter(fname, **kwargs)
    writer = tfrecord.TFRecordWriter(fname)
    size = 0
    for i, item in enumerate(samples):
        raw_data = map_func(item)
        size += len(raw_data["image"][0])
        writer.write(raw_data)

        if i % 1000 == 0:
            print(f"[{datetime.datetime.now()}] complete to write {i:06d} samples to shard {fname}")
    writer.close()
    print(f"[{datetime.datetime.now()}] complete to write samples to shard {fname}!!!")
    create_index(fname, fname+".idx")
    print(f"[{datetime.datetime.now()}] complete tfrecord2idx to shard {fname}!!!")
    return size


def main(dataset_root, dataset_split_root, dest, pattern, num_shards, num_workers):
    items = []
    dataset = ImageFolder(root=dataset_split_root,  loader=lambda x: x)
    for i in range(len(dataset)):
        path, class_idx = dataset[i]
        relpath = os.path.relpath(path, dataset_root)
        items.append((path, relpath, class_idx))

    def map_func(item):
        path, relpath, class_idx = item
        with open(os.path.join(path), "rb") as stream:
            image = stream.read()

        sample = {
            # "fname": (bytes(os.path.splitext(os.path.basename(name))[0], "utf-8"), "byte"),
            "metadata": (
                jsonpack(dict(path=relpath), maxlen=128),
                "byte"
            ),
            "image": (image, "byte"),
            "label": (class_idx, "int")
        }
        return sample
    
    os.makedirs(dest, exist_ok=False)
    make_wds_shards(
        pattern=os.path.join(dest, pattern),
        num_shards=num_shards,
        num_workers=num_workers,
        samples=items,
        map_func=map_func,
    )


if __name__ == "__main__":
    source = os.path.expanduser("/home/chenyaofo/datasets/imagenet")
    dest = os.path.expanduser("/home/chenyaofo/datasets/imagenet-tfrec")
    main(
        dataset_root=source,
        dataset_split_root=os.path.join(source, "train"),
        dest=os.path.join(dest, "train"),
        pattern="%06d.tfrecord",
        num_shards=1024,
        num_workers=8
    )
    main(
        dataset_root=source,
        dataset_split_root=os.path.join(source, "val"),
        dest=os.path.join(dest, "val"),
        pattern="%06d.tfrecord",
        num_shards=256,
        num_workers=8
    )

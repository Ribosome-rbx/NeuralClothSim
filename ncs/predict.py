import os
import sys
import argparse
import tensorflow as tf
import numpy as np
import shutil

from dataset.data import Data
from model.ncs import NCS
from utils.config import MainConfig
from utils.IO import writePC2Frames, readPC2
from global_vars import CHECKPOINTS_DIR, RESULTS_DIR

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def main(config, w=1.0):
    # Load model
    model = NCS(config)
    model.load_weights(os.path.join(CHECKPOINTS_DIR, config.name))

    # Init data
    data = Data(config, mode="test")

    # Predict & store
    print("Predicting...")
    folder = os.path.join(RESULTS_DIR, config.name)
    if os.path.isdir(folder): shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)

    for i, batch in enumerate(data):
        take = data.takes[i].split('/')[-1][:-4]
        take_folder = os.path.join(folder, take)
        os.makedirs(take_folder, exist_ok=True)
        # Filenames for results
        filenames = {
            "body": os.path.join(take_folder, "body.pc2"),
            "cloth": os.path.join(take_folder, config.garment.name + ".pc2"),
            "cloth_unskinned": os.path.join(take_folder, config.garment.name + "_unskinned.pc2"),
        }
        sys.stdout.write("\r" + str(i + 1) + "/" + str(len(data)))
        sys.stdout.flush()
        body, cloth, unskinned = model.predict(batch, w=w)
        # Store results
        writePC2Frames(filenames["body"], body.numpy())
        writePC2Frames(filenames["cloth"], cloth.numpy())
        writePC2Frames(filenames["cloth_unskinned"], unskinned.numpy())

        # transfer pc2 into npy
        for f in ["body", "cloth"]:
            file = readPC2(filenames[f])
            np.save(filenames[f].replace(".pc2", ".npy"), file['V'])
        print("pc2 to npy --> Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    # parser.add_argument("--gpu_id", type=str, default="")
    parser.add_argument("--motion", type=float, default=1.0)
    opts = parser.parse_args()

    folder_name = opts.folder
    type =  folder_name.split("_")[-1].lower()



    # Set GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0' # opts.gpu_id

    # Limit VRAM usage
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if not gpus:
        print("No GPU detected")
        sys.exit()

    config = MainConfig('configs/smplx.json') # opts.config
    config.name = folder_name
    config.body.model = folder_name
    config.garment.name = type
    config.data.fps = 30
    main(config, w=opts.motion)

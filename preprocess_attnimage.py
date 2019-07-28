# -*- coding: utf-8 -*-
import argparse
import pickle
import os
import subprocess
from miscc.utils import mkdir_p


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename_info", type=str,
                        default='../AttnGAN/data/birds/test/filenames.pickle')
    parser.add_argument("--attn_data", type=str,
                        default="../AttnGAN/output/birds_attn2_2019_07_25_10_29_58/Model/netG_epoch_600/valid/single")
    parser.add_argument("--output", type=str,
                        default="./datasets/bird/attn")
    parser.add_argument("--captions_per_image", type=int, default=10)
    opt = parser.parse_args()

    with open(opt.filename_info, "rb") as f:
        filename_info = pickle.load(f)
        print("Load from:", opt.filename_info)

    if not os.path.isdir(opt.output):
        print('Make a new folder: ', opt.output)
        mkdir_p(folder)

    serialized_index = 0
    for f_info in filename_info:
        for index in range(opt.captions_per_image):
            filename = os.path.join(opt.attn_data, "{}_s-1_sent{}.jpg".format(f_info, index))
            assert os.path.isfile(filename), "ERROR: invelid file name {}".format(filename)
            output = os.path.join(opt.output, "{}.jpg".format(serialized_index))
            serialized_index += 1
            exit_code = subprocess.check_call(["cp", filename, output])
            assert exit_code == 0,\
                "ERROR: cp process failed. code({}). cp from {}, to {}".format(exit_code, filename, output)

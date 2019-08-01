# -*- coding: utf-8 -*-
import os
import pickle
import argparse
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import ImageFilter, Image


def load_bbox(data_dir):
    bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
    df_bounding_boxes = pd.read_csv(bbox_path,
                                    delim_whitespace=True,
                                    header=None).astype(int)
    filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
    df_filenames = \
        pd.read_csv(filepath, delim_whitespace=True, header=None)
    filenames = df_filenames[1].tolist()
    print('Total filenames: ', len(filenames), filenames[0])
    filename_bbox = {img_file[:-4]: [] for img_file in filenames}
    for i in range(0, len(filenames)):
        bbox = df_bounding_boxes.iloc[i][1:].tolist()
        key = filenames[i][:-4]
        filename_bbox[key] = bbox
    return filename_bbox


def pre_transform(pil_image, imsize, bbox):
    width, height = pil_image.size
    r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
    center_x = int((2 * bbox[0] + bbox[2]) / 2)
    center_y = int((2 * bbox[1] + bbox[3]) / 2)
    y1 = np.maximum(0, center_y - r)
    y2 = np.minimum(height, center_y + r)
    x1 = np.maximum(0, center_x - r)
    x2 = np.minimum(width, center_x + r)
    pil_image = pil_image.crop([x1, y1, x2, y2])
    return pil_image.resize((imsize, imsize))


def get_gaussian_blur_transform(imsize):
    image_transform = transforms.Compose([
        transforms.Resize(imsize),
        transforms.Lambda(lambda im: im.filter(ImageFilter.GaussianBlur(radius=imsize/128))),
        transforms.Grayscale(num_output_channels=3)])
    return image_transform


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--attn_ref_data_path", type=str,
                        default='../AttnGAN/data/birds/')
    parser.add_argument("--target_path", type=str,
                        default="./datasets/birds_gaussianblur")
    parser.add_argument("--imsize", type=float,
                        default=256)
    opt = parser.parse_args()

    B_transform = get_gaussian_blur_transform(opt.imsize)

    # Load files
    with open(os.path.join(opt.attn_ref_data_path, "train/filenames.pickle"), "rb") as f:
        train_filenames = pickle.load(f)
    with open(os.path.join(opt.attn_ref_data_path, "test/filenames.pickle"), "rb") as f:
        test_filenames = pickle.load(f)
    filename2bbox = load_bbox(opt.attn_ref_data_path)

    print("for test data")
    for idx, test_file in enumerate(test_filenames):
        # Print progress
        if idx % 500 == 0:
            print("Processing {}th image ...".format(idx))

        # Load image
        image_path = os.path.join(opt.attn_ref_data_path, "CUB_200_2011/images", test_file+".jpg")
        image = Image.open(image_path).convert("RGB")

        # Preprocess image
        B_image = pre_transform(image, opt.imsize, filename2bbox[test_file])
        A_image = B_transform(B_image)
        AB_image = Image.new("RGB", (opt.imsize*2, opt.imsize))
        AB_image.paste(A_image, (0, 0))
        AB_image.paste(B_image, (opt.imsize, 0))

        # Save preprocessed image
        AB_path = os.path.join(opt.target_path, "test", "{}.jpg".format(idx))
        AB_image.save(AB_path)

    print("for train data")
    for idx, train_file in enumerate(train_filenames):
        # Print progress
        if idx % 500 == 0:
            print("Processing {}th image ...".format(idx))

        # Load image
        image_path = os.path.join(opt.attn_ref_data_path, "CUB_200_2011/images", train_file+".jpg")
        image = Image.open(image_path).convert("RGB")

        # Preprocess image
        B_image = pre_transform(image, opt.imsize, filename2bbox[train_file])
        A_image = B_transform(B_image)
        AB_image = Image.new("RGB", (opt.imsize*2, opt.imsize))
        AB_image.paste(A_image, (0, 0))
        AB_image.paste(B_image, (opt.imsize, 0))

        # Save preprocessed image
        AB_path = os.path.join(opt.target_path, "train", "{}.jpg".format(idx))
        AB_image.save(AB_path)

    print("Finish.")

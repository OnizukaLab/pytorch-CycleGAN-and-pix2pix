#!/usr/bin/env bash
ATTN_PATH=${1:-../AttnGAN/output/birds_attn2_2019_07_25_10_29_58/Model/netG_epoch_550/valid/single}
PHASE=${2:-attn}

# Prepare AttnGAN image for pix2pix
python preprocess_attnimage.py\
 --attn_data ${ATTN_PATH}

# Colorization
python test.py --dataroot ./datasets/bird --name color_pix2pix --model colorization\
 --phase ${PHASE} --name color_pix2pix --gpu_id 0 --num_test 30000

# Prepare for fid_score
mkdir ./datasets/bird/${PHASE}/fake_images
for i in `seq 0 9`
do
 cp /opt/project/pytorch-CycleGAN-and-pix2pix/results/color_pix2pix/${PHASE}_latest/images/${i}*_fake*\
  /opt/project/pytorch-CycleGAN-and-pix2pix/results/color_pix2pix/${PHASE}_latest/fake_images/
done
rm -rf /opt/project/pytorch-CycleGAN-and-pix2pix/results/color_pix2pix/${PHASE}_latest/images

# Calculate FID
cd /opt/pytorch-fid/
python fid_score.py /opt/project/pytorch-CycleGAN-and-pix2pix/datasets/bird/test\
# /opt/project/pytorch-CycleGAN-and-pix2pix/results/color_pix2pix/${PHASE}_latest/fake_images -c 0

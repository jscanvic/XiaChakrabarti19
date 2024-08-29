#!/usr/bin/env python3
import argparse
import glob

import tensorflow as tf
import numpy as np
from utils import utils as ut
from utils import pix as net, pix_blind as net_blind
from imageio import imsave, imread

import os.path

def read_image(path):
    return np.float32(imread(path)) / 255.0

def save_image(path, img):
    img = np.maximum(0.,np.minimum(1.,img))
    img = np.uint8(img*255.)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imsave(path, img)

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='wts/blind', help='path to trained model')
parser.add_argument('--data', default='data/Test_data_Helen', help='path to data')
parser.add_argument('--outpath', default=None, help='where to save predictions')
parser.add_argument('--with_kernels', default=False, action='store_true', help='whether to save kernels or not')
opts = parser.parse_args()

if opts.path.endswith('.npz'):
    weights_path = opts.path
else:
    weights_path = opts.path
    checkpoints = ut.ckpter(weights_path + '/iter_*.model.npz')
    weights_path = checkpoints.latest

ground_truth_paths = glob.glob(f"{opts.data}/*_gt/*.png")
if len(ground_truth_paths) != 0:
    kernel_indices = [ (i,j) for i in range(1, 11) for j in range(13, 29, 2) ]
    blurry_image_paths_table = {}
    for ground_truth_path in ground_truth_paths:
        blurry_image_path_base = ground_truth_path.replace('.png', '').replace('_gt', '_blur')
        blurry_image_paths = [
            f"{blurry_image_path_base}_ker{i:02d}_blur_k{j}.png"
            for i, j in kernel_indices
        ]
        blurry_image_paths_table[ground_truth_path] = blurry_image_paths
    ground_truth_paths_table = {
        blurry_image_path: ground_truth_path
        for ground_truth_path, blurry_image_paths in blurry_image_paths_table.items()
        for blurry_image_path in blurry_image_paths
    }
else:
    blurry_image_paths = glob.glob(f"{opts.data}/*.png")
    ground_truth_paths_table = {
        blurry_image_path: None
        for blurry_image_path in blurry_image_paths
    }

#### Build Graph
blurry_image_path_placeholder = tf.placeholder(tf.string)
blurry_image_placeholder = tf.read_file(blurry_image_path_placeholder)
blurry_image_placeholder = tf.image.decode_png(blurry_image_placeholder, channels=3, dtype=tf.uint8)
blurry_image_placeholder = tf.to_float(blurry_image_placeholder) / 255.
blurry_images_placeholder = tf.stack([blurry_image_placeholder], axis=0)

is_training_placeholder = tf.placeholder_with_default(False, shape=[])
if opts.with_kernels:
    model = net_blind.Net(is_training_placeholder)
    deblurred_images_placeholder, kernel_estimates_placeholder = model.generate(blurry_images_placeholder)
else:
    model = net.Net(is_training_placeholder)
    deblurred_images_placeholder = model.generate(blurry_images_placeholder)
    kernel_estimates_placeholder = None
deblurred_images_placeholder = blurry_images_placeholder + deblurred_images_placeholder

# Create session
session = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=4))
session.run(tf.global_variables_initializer())

# Load model
print(f"Restoring model from {weights_path}")
ut.loadNet(weights_path, model.weights, session)

print("Done!")

mse_accumulator = 0.
psnr_accumulator = 0.
N = 0
for counter, (blurry_image_path, ground_truth_path) in enumerate(ground_truth_paths_table.items()):
    if ground_truth_path is not None:
        ground_truth_image = read_image(ground_truth_path)
        ground_truth_image = ground_truth_image[np.newaxis]
    else:
        ground_truth_image = None

    feed_dict = {blurry_image_path_placeholder: blurry_image_path}
    if kernel_estimates_placeholder is not None:
        _, deblurred_images, kernel_estimates = session.run([
            blurry_images_placeholder,
            deblurred_images_placeholder,
            kernel_estimates_placeholder
        ], feed_dict=feed_dict)
    else:
        _, deblurred_images = session.run([
            blurry_images_placeholder,
            deblurred_images_placeholder
        ], feed_dict=feed_dict)
        kernel_estimates = None
    deblurred_image = deblurred_images[0]
    if kernel_estimates is not None:
        kernel_estimate = kernel_estimates[0]

    if ground_truth_image is not None:
        mse = np.mean((ground_truth_image-deblurred_images)**2, axis=(1,2,3))
        psnr = -10. * np.log10(mse)
        mse = np.mean(mse)
        psnr = np.mean(psnr)
        mse_accumulator += mse
        psnr_accumulator += psnr
        N += 1

    if opts.outpath is not None:
        deblurred_image_filename = os.path.basename(blurry_image_path)
        path = f"{opts.outpath}/{deblurred_image_filename}"
        save_image(path, deblurred_image)

        if opts.with_kernels:
            deblurred_image_filename = os.path.basename(blurry_image_path)
            path = f"{opts.outpath}/kernels/{deblurred_image_filename}"
            save_image(path, kernel_estimate / kernel_estimate.max())

    print(f"Finsih {counter}/{len(ground_truth_paths_table)} images")

if N != 0:
    mse = mse_accumulator / N
    psnr = psnr_accumulator / N
    print(f"MSE: {mse:.4f}")
    print(f"PSNR: {psnr:.2f}")

#!/usr/bin/env python3

import os
import argparse
from PIL import Image
import numpy as np
from numpy.linalg import norm
from numpy import linalg
from random import normalvariate
from math import sqrt

from power_method import power_method_svd
from householder_method import householder_svd

def compress_image(input_path, output_path, svd_method, N):
    # Load image
    img = Image.open(input_path)
    m, n = img.size
    k = int(max(1, ((-(m+n) + sqrt((m+n)**2 + 4 * m * n / 9 / N)) / 2)))
    print(k)
    
    # Convert image to array
    img_array = np.array(img, dtype='float64' )
    
    # Separate RGB channels
    red_channel = img_array[:,:,0]
    green_channel = img_array[:,:,1]
    blue_channel = img_array[:,:,2]

    # Apply SVD to each channel
    U_r, S_r, Vt_r = svd_method(red_channel)
    U_g, S_g, Vt_g = svd_method(green_channel)
    U_b, S_b, Vt_b = svd_method(blue_channel)

    #print(U_r.shape, S_r.shape, Vt_r.shape)

    # Truncate singular values
    S_r_trunc = np.diag(S_r[:k])
    S_g_trunc = np.diag(S_g[:k])
    S_b_trunc = np.diag(S_b[:k])

    np.savez(output_path,
    U_r=U_r[:,:k], S_r=S_r_trunc, Vt_r=Vt_r[:k,:],
    U_g=U_g[:,:k], S_g=S_g_trunc, Vt_g=Vt_g[:k,:],
    U_b=U_b[:,:k], S_b=S_b_trunc, Vt_b=Vt_b[:k,:])

def decompress_intermediate(input_path, output_path):
    data = np.load(input_path)
    U_r, S_r, Vt_r = data['U_r'], data['S_r'], data['Vt_r']
    U_g, S_g, Vt_g = data['U_g'], data['S_g'], data['Vt_g']
    U_b, S_b, Vt_b = data['U_b'], data['S_b'], data['Vt_b']
    data.close()

    # Reconstruct compressed channels
    red_channel_compressed = U_r @ S_r @ Vt_r
    green_channel_compressed = U_g @ S_g @ Vt_g
    blue_channel_compressed = U_b @ S_b @ Vt_b

    # Stack channels together
    compressed_img_array = np.stack(
        (red_channel_compressed, green_channel_compressed, blue_channel_compressed), 
        axis=-1
    )

    # Convert array to image and save
    decompressed_img = Image.fromarray(np.uint8(compressed_img_array))
    decompressed_img.save(output_path)
    

def parse_arguments():
    parser = argparse.ArgumentParser(description="Image compression and decompression using SVD")
    parser.add_argument("--mode", choices=["compress", "decompress"], required=True, help="Mode: compress or decompress")
    parser.add_argument("--method", choices=["numpy", "simple", "advanced"], help="SVD method (only for compress mode)")

    parser.add_argument("--compression", type=int, help="Compression factor (only for compress mode)")
    parser.add_argument("--in_file", required=True, help="Path to the input file")
    parser.add_argument("--out_file", required=True, help="Path to the output file")

    args = parser.parse_args()

    if args.mode == "compress":
        if not args.method:
            parser.error("Method argument is required in compress mode")
        if not args.compression:
            parser.error("Compression factor argument is required in compress mode")
    elif args.mode == 'decompress':
        if args.method or args.compression:
            parser.error("Method and compression arguments are not applicable in decompress mode")
    
    return args


if __name__ == "__main__":
    args = parse_arguments()
    if args.mode == "compress":
        svd_fs = {
            "numpy" : lambda m : np.linalg.svd(m, full_matrices=False),
            "simple" : power_method_svd,
            "advanced" : householder_svd
        }
        compress_image(args.in_file, args.out_file, svd_fs[args.method], args.compression)
        image_size = os.path.getsize(args.in_file)
        print(image_size / os.path.getsize(args.out_file))
    else:
        decompress_intermediate(args.in_file, args.out_file)


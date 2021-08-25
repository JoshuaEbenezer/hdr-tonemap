import cv2
import time
from scipy.io import savemat
import array
import colour
import numpy as np
import hdr_utils
import argparse
import os
import OpenEXR
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Run a TMO on a YUV HDR file')
parser.add_argument('input_file',help='File containing input yuv HDR video')
parser.add_argument('tonemap_method',help='Tonemapping method')
parser.add_argument('framenos',help='Number of frames in video')
parser.add_argument('exposure',help='TMO exposure')
args = parser.parse_args()

tonemap_method = args.tonemap_method
input_file =args.input_file 
framenos = int(args.framenos)
exposure=float(args.exposure)
h = 2160
w = 3840

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(os.path.basename(input_file)[:-4]+'_'+tonemap_method+'.mp4',fourcc, 30.0, (w,h))
for frame_num in range(framenos):
    start= time.time()
    y,u,v= hdr_utils.hdr_yuv_read(input_file,frame_num,h,w)
    yuvend = time.time()
    print(yuvend-start,' is YUV read time')

    # concatenate YUV
    yuv = np.stack((y,u,v),axis=2)

    # linearize YUV
    yuv_linear = colour.models.eotf_PQ_BT2100(rgb_bt2020_pq)

    # convert to RGB in BT2020
    rgb_bt2020_linear = hdr_utils.yuv2rgb_bt2020(yuv_linear[:,:,0],yuv_linear[:,:,1],yuv_linear[:,:,2])

    # remove out of gamut colors and clip
    # clip luminance to 300 nits --> heuristic, use if it improves appearance
    # rgb_bt2020_linear_clipped = np.clip(rgb_bt2020_linear,0,300)

    # convert to BT709
    rgb_bt709_linear = colour.RGB_to_RGB(rgb_bt2020_linear,colour.models.RGB_COLOURSPACE_BT2020,colour.models.RGB_COLOURSPACE_BT709)
    rgb_bt709_linear_clipped = np.clip(rgb_bt709_linear,0,100)

    # apply tonemap
    rgb_bt709_tonemapped_linear = hdr_utils.tonemap(rgb_bt709_linear_clipped,tonemap_method,exposure)

    rgb_bt709_tonemapped_gamma = colour.models.oetf_BT709(rgb_bt709_tonemapped_linear) 
    rgb_bt709_tonemapped_gamma = (rgb_bt709_tonemapped_gamma*255).astype(np.uint8)
    end = time.time()
    print(end-yuvend,' is colour time')
    bgr_bt709_tonemapped_gamma = np.stack((rgb_bt709_tonemapped_gamma[:,:,2],rgb_bt709_tonemapped_gamma[:,:,1],rgb_bt709_tonemapped_gamma[:,:,0]),2)

    out.write(bgr_bt709_tonemapped_gamma)


out.release()

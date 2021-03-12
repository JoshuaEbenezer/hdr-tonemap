import cv2
from joblib import Parallel,delayed
import pandas as pd
from scipy.stats import gmean,cumfreq
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
parser.add_argument('csv_file',help='File containing input yuv HDR video names, fps, resolutions')
parser.add_argument('yuv_folder',help='Folder containing YUV files')
parser.add_argument('out_folder',help='Folder for output')
args = parser.parse_args()

csv_file = pd.read_csv(args.csv_file) 
names = csv_file["names"]
fps_list = csv_file["fps"]
res = csv_file["res"]
framenos_list = csv_file["framenos"]
yuv_folder = args.yuv_folder
out_folder = args.out_folder

alpha_A = 0.98
alpha_B = 0.98
alpha_a = 0.98

nbins = 5000

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

def vid_tmo(i):
    f = names[i]
    basename = os.path.basename(f)
    input_file = os.path.join(yuv_folder,basename[:-3]+'yuv')
    w = int(res[i][0:4])
    h = int(res[i][-4:])
    num,denom = fps_list[i].split('/')
    fps = float(num)/float(denom)
    framenos = framenos_list[i]
    output_file = os.path.join(out_folder,os.path.basename(input_file)[:-4]+'_vidtmo.mp4')
    # if(os.path.exists(output_file)):
    #     return
    out = cv2.VideoWriter(output_file,fourcc, fps, (w,h))
    max_C = h*w
    beta = 0.999
    B_upper = beta*max_C
    B_lower = (1-beta)*max_C

    print(framenos)
    for frame_num in range(framenos):
        y,u,v= hdr_utils.hdr_yuv_read(input_file,frame_num,h,w)

        # convert to RGB in BT2020
        rgb_bt2020_pq = hdr_utils.yuv2rgb_bt2020(y,u,v)
        rgb_bt2020_pq = np.clip(rgb_bt2020_pq.astype(np.float32),0,1024)/1024.0

        # linearize
        rgb_bt2020_linear = colour.models.eotf_PQ_BT2100(rgb_bt2020_pq)

        # clip luminance to 100 nits
#        rgb_bt2020_linear_clipped = np.clip(rgb_bt2020_linear,0,300)

        # convert to BT709
        rgb_bt709_linear = colour.RGB_to_RGB(rgb_bt2020_linear,colour.models.RGB_COLOURSPACE_BT2020,colour.models.RGB_COLOURSPACE_BT709)

        # find linear luminance
        y_bt709_linear = 0.2126*rgb_bt2020_linear[:,:,0]+0.7152*rgb_bt2020_linear[:,:,1]+0.0722*rgb_bt2020_linear[:,:,2] 

        # find cumfreq
        C,lowerlimit,binsize,_ = cumfreq(y_bt709_linear.flatten(),nbins)

        # define x axis of cumfreq
        hdr_range = np.linspace(lowerlimit,np.max(rgb_bt2020_linear),nbins)
        B_upper_index =np.searchsorted(C,B_upper) 
        B_lower_index =np.searchsorted(C,B_lower) 
        
        # clamping vals
        hdr_lower = hdr_range[B_lower_index]
        hdr_upper = hdr_range[B_upper_index]
        
        y_bt709_linear_clipped = np.clip(y_bt709_linear,hdr_lower,hdr_upper)
        
        y_bt709_linear_clipped = y_bt709_linear_clipped-hdr_lower+1
    #    y_bt709_linear_clipped = (y_bt709_linear_clipped-hdr_lower)*255/(hdr_upper-hdr_lower)

        # find current parameters for frame
        L_av = gmean(y_bt709_linear_clipped,None)
        Lmax = np.max(y_bt709_linear_clipped)
        Lmin = np.min(y_bt709_linear_clipped)
        A = Lmax-L_av
        B = L_av-Lmin
        a = 0.18*2**(2*(B-A)/(B+A))

        if(frame_num>=1):
            # update smooth parameters
            A_curr = (1-alpha_A)*A_prev+alpha_A*A
            B_curr = (1-alpha_B)*B_prev+alpha_B*B
            a_curr = (1-alpha_a)*a_prev+alpha_a*a
        else:
            A_curr = A
            B_curr = B
            a_curr = a

        print(L_av)
        
        # apply tonemap
        lwhite_sq = hdr_upper**2 
        image = a_curr*y_bt709_linear_clipped/L_av
        y_bt709_tonemapped_linear = image*(1+image/lwhite_sq)/(1+image)
        print(np.min(y_bt709_tonemapped_linear),np.max(y_bt709_tonemapped_linear))

        print(A_curr)

        # update
        A_prev = A_curr
        B_prev = B_curr
        a_prev = a_curr

        rgb_bt709_tonemapped_linear = rgb_bt709_linear*np.expand_dims(y_bt709_tonemapped_linear/(y_bt709_linear+1e-5),2)
        rgb_bt709_tonemapped_gamma = colour.models.oetf_BT709(rgb_bt709_tonemapped_linear) 
        rgb_bt709_tonemapped_gamma = (rgb_bt709_tonemapped_gamma*255).astype(np.uint8)
        bgr_bt709_tonemapped_gamma = rgb_bt709_tonemapped_gamma[:, :, ::-1]

        out.write(bgr_bt709_tonemapped_gamma)


    out.release()
    return
print(len(names))
#_ = Parallel(n_jobs=-10)(delayed(vid_tmo)(split_no) for split_no in range(len(names)))
for i in range(len(names)):
    vid_tmo(i)

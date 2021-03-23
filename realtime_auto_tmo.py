import cv2
import skvideo.io
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
parser.add_argument('out_folder',help='Folder for output')
args = parser.parse_args()

csv_file = pd.read_csv(args.csv_file) 
names = csv_file["yuv"]
fps_list = csv_file["fps"]
framenos_list = csv_file["framenos"]
w_list = csv_file["w"]
h_list = csv_file["h"]
out_folder = args.out_folder

alpha_A = 0.98
alpha_B = 0.98
alpha_a = 0.98

nbins = 5000


def vid_tmo(i):
    input_file = names[i]
    basename = os.path.basename(input_file)

    crf = 10
    w = w_list[i] 
    h = h_list[i] 
    fps = fps_list[i] 
    framenos = framenos_list[i]
    output_file = os.path.join(out_folder,os.path.basename(input_file)[:-4]+'_vidtmo.mp4')
    if(os.path.exists(output_file)):
        return
    inputdict = {"-r":str(fps),"-s:v":str(w)+'x'+str(h),"-pix_fmt":"rgb24","-f":"rawvideo"}
    outputdict = {"-c:v":"libx264","-crf":str(crf),"-pix_fmt":"yuv420p"}
    writer = skvideo.io.FFmpegWriter(output_file,inputdict,outputdict,verbosity=1)
    max_C = h*w
    beta = 0.999
    B_upper = beta*max_C
    B_lower = (1-beta)*max_C

    print(framenos)

    frame_num = 0
    while(True):
        if (frame_num==framenos):
            break
        print(frame_num, "is the current frame number")
        y,u,v= hdr_utils.hdr_yuv_read(input_file,frame_num,h,w)

        # convert to RGB in BT2020
        rgb_bt2020_pq = hdr_utils.yuv2rgb_bt2020(y,u,v)
#        plt.hist(y.astype(np.float32).flatten(),bins='auto')
#        plt.show()
        rgb_bt2020_pq = np.clip(rgb_bt2020_pq.astype(np.float32),0,1024)/1024.0

        # linearize
        rgb_bt2020_linear = colour.models.eotf_PQ_BT2100(rgb_bt2020_pq)

        # clip luminance to 100 nits
#        rgb_bt2020_linear = np.clip(rgb_bt2020_linear,0,300)


        # find linear luminance
        y_bt2020_linear = 0.2627*rgb_bt2020_linear[:,:,0]+0.6780*rgb_bt2020_linear[:,:,1]+0.0593*rgb_bt2020_linear[:,:,2] 

        # find cumfreq
        C,lowerlimit,binsize,_ = cumfreq(y_bt2020_linear.flatten(),nbins)

        # define x axis of cumfreq
        hdr_range = np.linspace(lowerlimit,np.max(y_bt2020_linear),nbins)
        B_upper_index =np.searchsorted(C,B_upper) 
        B_lower_index =np.searchsorted(C,B_lower) 
        
        # clamping vals
        hdr_lower = hdr_range[B_lower_index]
        hdr_upper = hdr_range[B_upper_index]
        
        y_bt2020_linear_clipped = np.clip(y_bt2020_linear,hdr_lower,hdr_upper)
        
        y_bt2020_linear_clipped = y_bt2020_linear_clipped-hdr_lower+1
    #    y_bt2020_linear_clipped = (y_bt2020_linear_clipped-hdr_lower)*255/(hdr_upper-hdr_lower)

        # find current parameters for frame
        L_av = gmean(y_bt2020_linear_clipped,None)
        Lmax = np.max(y_bt2020_linear_clipped)
        Lmin = np.min(y_bt2020_linear_clipped)
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

        
        # apply tonemap
        lwhite_sq = hdr_upper**2 
        image = a_curr*y_bt2020_linear_clipped/L_av
        y_bt2020_tonemapped_linear = image*(1+image/lwhite_sq)/(1+image)


        # update
        A_prev = A_curr
        B_prev = B_curr
        a_prev = a_curr

        rgb_bt2020_tonemapped_linear = rgb_bt2020_linear*np.expand_dims(y_bt2020_tonemapped_linear/(y_bt2020_linear+1e-5),2)

        # convert to BT709
#        rgb_bt709_tonemapped_linear = colour.RGB_to_RGB(rgb_bt2020_tonemapped_linear,colour.models.RGB_COLOURSPACE_BT2020,colour.models.RGB_COLOURSPACE_BT709)
#        rgb_bt709_tonemapped_gamma = colour.models.oetf_BT709(rgb_bt709_tonemapped_linear) 
        rgb_bt2020_tonemapped_gamma = colour.models.oetf_BT2020(rgb_bt2020_tonemapped_linear)
        rgb_tonemapped_gamma = (rgb_bt2020_tonemapped_gamma*255).astype(np.uint8)

        writer.writeFrame(rgb_tonemapped_gamma)
        frame_num = frame_num+1

    writer.close()
    
    return
print(len(names))
#_ = Parallel(n_jobs=-10)(delayed(vid_tmo)(split_no) for split_no in range(len(names)))
for i in range(len(names)):
    vid_tmo(i)

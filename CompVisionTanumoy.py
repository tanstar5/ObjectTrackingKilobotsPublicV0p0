"""
Updated on Fri Mar 11 12:09:29 2022
Library for Kilobots tracking software using OpenCV, matplotlib, scipy library
@author: Tanumoy Saha
"""

# Imports _____________________________________________________________________
import cv2 as cv
import sys
import numpy as np #used to create array and manipulate them
import scipy as sp #used for using sparse matrix methods for big matrices
from scipy import sparse
import matplotlib as mt #used for reading images as matrix/arrays and plotting them
import time
from Tracking_Objs import Tracking_Objs
#from Tracking_Objs import plot_analysis_results
from Tracking_Objs import *


# Tracking parameters
read_from_cam = 0 # 0 if from video file

webcam_num = 0;
vid_file = 'original_test.avi' 
resize_frame = 0;


write_vid = 1;
out1 = cv.VideoWriter('original_testRes.avi',cv.VideoWriter_fourcc(*'MJPG'),10, (640,480))
out2 = cv.VideoWriter('camfeed_testRes.avi',cv.VideoWriter_fourcc(*'MJPG'),10, (640,480))
out3 = cv.VideoWriter('analysis_testRes.avi',cv.VideoWriter_fourcc(*'MJPG'),10, (640,480))

Num_frames_to_keep_in_heap = 20;
write_tracking_analysis_to_file = 0;


#fig, axes = mt.pyplot.subplots(nrows = 2, ncols = 2, sharex=True, sharey = True);
# FUNCTION DEFINATIONS_________________________________________________________
#_________________Load functions_______________________________________________
def list_ports():
    """
    Copied from: https://stackoverflow.com/questions/57577445/list-available-cameras-opencv-python
    Test the ports and returns a tuple with the available ports and the ones that are working.
    """
    non_working_ports = []
    dev_port = 0
    working_ports = []
    available_ports = []
    while len(non_working_ports) < 6: # if there are more than 5 non working ports stop the testing. 
        camera = cv.VideoCapture(dev_port)
        if not camera.isOpened():
            non_working_ports.append(dev_port)
            print("Port %s is not working." %dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                working_ports.append(dev_port)
            else:
                print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
                available_ports.append(dev_port)
        dev_port +=1
    return available_ports,working_ports,non_working_ports

def load_image(filename):
    """loading image using base library function
    matplotlib.pyplot.imread(fname, format=None)
    Takes the filepath or filename as argument; Returns the image abd image size 
    as a tuple image,img_size"""
    image = mt.pyplot.imread(filename) #matplotlib.pyplot.imread(fname, format=None)
    #image = scipy.misc. #matplotlib.pyplot.imread(fname, format=None)
    img_size = image.shape
    return image,img_size

def load_image_stack(filepath,load_image_to_mem = 1):
    """loading image (.tiff) using base library function
    matplotlib.pyplot.imread(fname, format=None)
    Takes the filepath as argument; Returns the image_stack and list of filepaths 
    as a tuple image,img_size"""
    files = os.listdir(filepath)
    img_paths = [None]*len(files)    
    for i in range(0,len(files)):
        # check if its an image
        if files[i].endswith(('.tif')):
            img_paths[i] = filepath +'/'+ files[i]
            
    image_first = mt.pyplot.imread(img_paths[0])
    image_size = image_first.shape
    num_images = len(files)
    array_size = np.append(image_size,num_images)
    image_stack_array = np.full(array_size,None)
    if (load_image_to_mem == 1):        
        for i in range(0,len(files)):
            # check if its an image
            if files[i].endswith(('.tif')):
                #img_paths[i] = filepath +'/'+ files[i]
                image_current =  mt.pyplot.imread(img_paths[i])
                image_stack_array[:,:,i] = image_current
        return img_paths, image_stack_array
    else:
        return  img_paths, None
#_________________Image/frame pre-processing functions_________________________
def contrast_image(image,contrast = 20,method = 'minmax'):
    """Loads image as matrix; Takes Contrast value from 1-100;
    method=['mean','median','minmax']; returns corrected image,contrast curve, 
    8bit intensity values"""
    image_current = image
    # brightness contrast correction
    #image_current = image_current - np.min(image_current) 
    image_current_normed = image_current
    image_current_normed = image_current_normed - np.min(image_current_normed);
    image_current_BCcorrected = image_current_normed;
    if method == 'mean':
        avg_intensity = np.mean(image_current)
        image_current_BCcorrected =  255/(np.exp(-contrast/100*(image_current_normed - avg_intensity) ) +1)
    elif(method== 'median'):
        avg_intensity = np.median(image_current)
        image_current_BCcorrected =  255/(np.exp(-contrast/100*(image_current_normed - avg_intensity) ) +1)
    elif(method == 'minmax'):
        #avg_intensity = (np.max(image_current) + np.min(image_current))/2
        avg_intensity = np.max([30, np.mean(image_current_normed[image_current_normed>0])])
        image_current_BCcorrected =  255/(np.exp(-contrast/100*(image_current_normed - avg_intensity) ) +1)
    elif(method == 'scale'):
        avg_intensity = np.min(image_current_normed[image_current_normed>0])
        image_current_BCcorrected =  image_current_normed/(np.max(image_current_normed))*255;
        
    else:
        print('method not defined')
        avg_intensity = 125
    
    intensity_vals = np.linspace(0, 255, 256)    
    contrast_line = 255/(np.exp(-contrast/100*(intensity_vals - avg_intensity))+1)
    return image_current_BCcorrected.astype(np.uint8), contrast_line,  intensity_vals

def color2gray(frame,channel_amplify=np.array([1,1,1])):
    channel_amplify = channel_amplify*1.0;
    channel_amplify = channel_amplify/np.sum(channel_amplify)
    frameP = frame[:,:,2]*channel_amplify[0] + frame[:,:,1]*channel_amplify[1] +\
        frame[:,:,0]*channel_amplify[2]
        
    frameRGB = frame;#0->B; 1->G; 2->R
    frameRGB[:,:,0] =  frame[:,:,2]*channel_amplify[0];
    frameRGB[:,:,1] =  frame[:,:,1]*channel_amplify[1];
    frameRGB[:,:,2] =  frame[:,:,0]*channel_amplify[2];
    return frameP.astype(np.uint8), frameRGB.astype(np.uint8)
    
def frame_preprocessing(frame,channel_amplify=np.array([.3,.6,.1])):
    #frame = cv.cvtColor(frame[:,:,channel], cv.COLOR_RGB2HSV)
    frame_copy = frame.copy();
    frameP, frameRGB= color2gray(frame_copy,channel_amplify)
    #ret,frameP = cv.threshold(frameP,127,255,cv.THRESH_BINARY)
    #frameP,tmp1,tmp2 = contrast_image(frameP,contrast = -50,method = 'median')
    return frameP.astype(np.uint8),frameRGB.astype(np.uint8)

#_________________Tracking functions___________________________________________
# Algo for tracking
# Calc optical flow
def optical_flow(framePprev,framePcurrent,win_size=15,background_pix_displacement = 30,non_linearity = 10):
    flow = \
        cv.calcOpticalFlowFarneback(framePprev, framePcurrent, None, 0.5, 3, win_size, 2, 5, 1.2, 0);
    mag_flow, direction_flow = cv.cartToPolar(flow[:,:,0], flow[:,:,1]);   
    flow_normed = mag_flow;
    flow_normed[mag_flow>0] = flow_normed[mag_flow>0]/np.max(flow_normed[mag_flow>0])*255
    #min_non_zero =  np.min(flow_normed[flow_normed>0]);
    #mask,_unused,_unused = \
    #    contrast_image(flow_normed,contrast = 10,method = 'minmax');
    image_current_normed = flow_normed;
    image_current_normed = image_current_normed - np.min(image_current_normed);
    avg_intensity = \
        np.max([background_pix_displacement, np.mean(image_current_normed[image_current_normed>0])])
    mask_flow =  255/(np.exp(-non_linearity/100*(image_current_normed - avg_intensity) ) +1) 
    mask_flow[mask_flow<125] = 0; 
    mask_flow[mask_flow>=125] = 255; 
    return mag_flow, flow_normed.astype(np.uint8),mask_flow.astype(np.uint8)
# Generate Mask using flow field and edge image
def gen_edge_mask(gray_scale_image,blur_window = (5,5),low_thresh=50,high_thresh = 150):
    gray_scale_image_blur = cv.GaussianBlur(gray_scale_image,blur_window,0);####    
    detected_edges = cv.Canny(gray_scale_image_blur,low_thresh,high_thresh,L2gradient=True) 
    return detected_edges

def generate_moving_edges_mask(mask_flow,gray_scale_image,blur_window = (5,5),low_thresh=50,high_thresh = 150):
    detected_edges = gen_edge_mask(gray_scale_image,blur_window,low_thresh,high_thresh)
    moving_edges = cv.bitwise_and(detected_edges,mask_flow);       
    return moving_edges.astype(np.uint8),detected_edges.astype(np.uint8) 

def gen_mask_moving_objs(flow,thresh=125):
    mag_flow, direction_flow = cv.cartToPolar(flow[:,:,0], flow[:,:,1]);
    mag_flow = mag_flow/np.max(mag_flow)*255
    mag_flow_norm,_unused,_unused = \
        contrast_image(mag_flow,contrast = 20,method = 'minmax');
    thresh = np.max( [0,np.mean(mag_flow_norm[mag_flow_norm>0])] );
    print(np.min(mag_flow_norm),np.max(mag_flow_norm),thresh)
    mag_flow_norm[mag_flow_norm<=thresh] = 0;
    mag_flow_norm[mag_flow_norm>thresh] = 255;
    return mag_flow_norm,mag_flow
# Detect Contours for sparse objects and/or distance transform for dense objects
def detect_contours_and_bound_poly(mask_edges,imageRGB_to_display,min_rect_area=10,max_rect_area=10):
    contours, hierarchy = cv.findContours(image=mask_edges, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE);
    cv.drawContours(image=imageRGB_to_display, contours=contours, contourIdx=-1, color=(0,0,255), thickness=1, lineType=cv.LINE_AA);
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    boundRect_area = [None]*len(contours)
    centers = [None]*len(contours)
    radius = [None]*len(contours)
    filtered_index = [];
    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        area_c= np.abs(cv.contourArea(c) );
        cv.drawContours(imageRGB_to_display, contours_poly, i, (0,255,0))
        boundRect[i] = cv.boundingRect(contours_poly[i])
        boundRect_area[i] = boundRect[i][2]*boundRect[i][3];
        centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])
        if((boundRect_area[i]<max_rect_area and boundRect_area[i]>=min_rect_area) and np.abs(boundRect[i][2]-boundRect[i][3])<=5):
        #if((area_c<max_rect_area and area_c>=min_rect_area) and np.abs(boundRect[i][2]-boundRect[i][3])<=5):    
            cv.rectangle(imageRGB_to_display, (int(boundRect[i][0]), int(boundRect[i][1])), \
                         (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), (0,0,255), 2)
            filtered_index.append(i);    
        #
    return imageRGB_to_display,contours_poly,boundRect,np.array(centers),filtered_index

# Generate tracks/traces
def write_IDs_on_objs(newID_list,  new_coor_list,frame,color = (0, 200, 0),fontScale = .5,rad = 20):
    print('DEBUG: len(newID_list)',len(newID_list))
    for ind in range(len(newID_list)):
        org = (int(new_coor_list[ind,0]),int(new_coor_list[ind,1]));
        print('DEBUG org',org)
        font = cv.FONT_HERSHEY_SIMPLEX;        
        #color = (0, 200, 0);
        thickness = 1;
        print('Writing');
        frame = cv.putText(frame, 'ID:'+str(newID_list[ind]), org, font, fontScale, color, thickness, cv.LINE_AA);
        frame = cv.circle(frame, org, rad, color, thickness)
    return frame;     
    
    
# Cluster Analysis
  #write a function for neighbours per particle and plot histogram; genrate counts of neighbours vs squared velocity

    


#_________________Data Visualization functions_________________________________



#_________________File operarations____________________________________________




#RUNTIME CODE__________________________________________________________________
#______________________________________________________________________________
# source code library: https://docs.opencv.org/
if (read_from_cam == 1):
    print('MESSAGE: Reading from cam webcam no:', webcam_num);
    cap = cv.VideoCapture(webcam_num)
else:
    print('MESSAGE: Reading from video file: filename',vid_file);
    cap = cv.VideoCapture(vid_file)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
else:
    ret, frame = cap.read()
    if (read_from_cam == 0 and resize_frame==1):
        print('MESSAGE:Resizing the image since big resolution file');
        frame = cv.resize(frame, (int(frame.shape[1]/2),int(frame.shape[0]/2)), interpolation =cv.INTER_AREA);
    framePprev,_unused = frame_preprocessing(frame,np.array([0,1,0]));
    #framePprev = cv.blur(framePprev,(5,5))
#centers_prev = np.array([-1,-1],ndmin=2);
centers_prev = np.empty((0,2)); 
#centers_prev_id = np.array([-1]);
centers_prev_id = np.empty((0,0));  
radius = 1000;
tracking_objs_test =  Tracking_Objs();
timer = 0
start_time = time.time();
time_diff = 0; 
#fig = mt.pyplot.figure()  
  
while True:
    # Capture frame-by-frame    
    ret, frame = cap.read()
    if write_vid == 1:
        out1.write(frame);
    if (read_from_cam == 0 and resize_frame==1):
        print('MESSAGE:Resizing the image since big resolution file');
        frame = cv.resize(frame, (int(frame.shape[1]/2),int(frame.shape[0]/2)), interpolation =cv.INTER_AREA);
    print('DEBUG: Size =', frame.shape)
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")        
        break
    
    # Frame operations
    framePcurrent,_unused = frame_preprocessing(frame,np.array([0,1,0]));
    mag_flow, flow_normed,mask_flow = \
        optical_flow(framePprev,framePcurrent,win_size=10,background_pix_displacement = 30,non_linearity = 20)    
    mask_edges,detected_edges = generate_moving_edges_mask(mask_flow,framePcurrent,blur_window = (5,5),low_thresh=50,high_thresh = 150)
    contours, hierarchy = cv.findContours(image=mask_edges, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)    
    framePprev = framePcurrent;
    
    # Display the resulting frame    
    mask = np.zeros(frame.shape,dtype=np.uint8);
    mask[:,:,1] = 0*mask_edges.astype(np.uint8);
    _,_,boundRect,centers,filtered_index = detect_contours_and_bound_poly(mask_edges,frame,min_rect_area=1450,max_rect_area=1800)    
    mask[:,:,2] = np.floor(1*mask_edges)
    mask[:,:,0] = np.floor(0.5*detected_edges);
    
    # Tracking
    print('DEBUG time_diff',time_diff)
    if (len(filtered_index)>0):
        centers_current = centers[filtered_index,:];         
        #newID_list = newID_list[index_not_to_del];
        if len(centers_current[:,0])>0:
            #centers_current,IDs =  gen_track_data(centers_current,centers_prev,centers_prev_id,radius)
            newID_list,  new_coor_list = tracking_objs_test.detect_me_next_frame_and_update(centers_current,45,timer,time_diff)
            centers_prev = new_coor_list;
        else:
            newID_list,  new_coor_list = tracking_objs_test.detect_me_next_frame_and_update(centers_prev,45,timer,time_diff)
            centers_prev = new_coor_list;#centers_prev_id = IDs;q
    else:
        newID_list,  new_coor_list = tracking_objs_test.detect_me_next_frame_and_update(centers_prev,45,timer,time_diff)
        centers_prev = new_coor_list;
    timer = timer + 1
    if ( timer%Num_frames_to_keep_in_heap==0):
        print('MESSAGE:dumping to file:frame_no',sys.getsizeof(tracking_objs_test.__dict__))
        #ar,ds,dis = tracking_objs_test.track_analysis(num_analysis_frames_from_present=100);
        #plot_analysis_results(ar,ds,dis,fig, axes)
        #tracking_objs_test.kill_static_objects(Num_frames_to_keep_in_heap)
        tracking_objs_test.write_coors_tofile_and_dump_frames(1,write_tracking_analysis_to_file,"output1.json")
        
    else:
        print('MESSAGE:will dump coors again when timer%200==0 ',timer%Num_frames_to_keep_in_heap)    
    mask = write_IDs_on_objs(newID_list,  new_coor_list,mask,color = (0, 255, 0),fontScale = .5,rad = 20)
    frame = write_IDs_on_objs(newID_list,  new_coor_list,frame,color = (255, 0, 0),fontScale = .1,rad = 10)
    end_time = time.time();
    time_diff = float(format(end_time-start_time));
    start_time = time.time();
    FPS = str( round(1/time_diff,2) );  
    print('DEBUG:','FPS:'+FPS+'Hz')
    mask = cv.putText(mask, 'FPS:'+FPS+'Hz', (10,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,200), 1, cv.LINE_AA);
    frame= cv.putText(frame, 'FPS:'+FPS+'Hz', (10,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv.LINE_AA); 
    if write_vid == 1:
        out3.write(mask);
    cv.imshow('Analysis', mask)
    if write_vid == 1:
        out2.write(frame);    
    cv.imshow('CamFeed', frame)
    key_stroke = cv.waitKey(5)& 0xff;
    if key_stroke == ord('q'):
        tracking_objs_test.write_coors_tofile_and_dump_frames(0,write_tracking_analysis_to_file,"output1.json")
        break
    elif(key_stroke == ord('d')):
        print(centers_current);
    else:
        continue
    
    
        
# Everything done, releasing the capture
cap.release()
out1.release()
out2.release()
out3.release()
cv.destroyAllWindows()

# Post tracking analysis_______________________________________________________



#writing/uploading data to cluster_____________________________________________



#EXAMPLES/TEST SCRIPTS_________________________________________________________
#______________________________________________________________________________
# img = cv.imread('C:/Users/tanum/source/repos/PythonTensorFlow/resource/falseEarth.jpg')
# if img is None:
#     sys.exit("Could not read the image.")
# cv.imshow("Display window", img)
# k = cv.waitKey(0)
# if k == ord("s"):
#     cv.imwrite("starry_night.png", img) 
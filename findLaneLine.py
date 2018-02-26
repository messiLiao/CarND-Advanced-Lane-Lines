import os
import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt
import argparse
import time

try:
    import pickle
except:
    import cPickle as pickle

should_exit = False
should_pause = False

def abs_sobel_thresh(img, orient='x', sobel_kernel=5, thresh=(0, 255)):    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    sobel_abs = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*sobel_abs/np.max(sobel_abs))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel, dtype=np.uint8)
    thresh_min, thresh_max = thresh
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

def mag_thresh(image, sobel_kernel=5, mag_thresh=(0, 255)):    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude 
    magnitude = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*magnitude/np.max(magnitude))
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel, dtype=np.uint8)
    thresh_min, thresh_max = mag_thresh
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

def dir_threshold(image, sobel_kernel=5, thresh=(0, np.pi/2)):    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobel_x = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    sobel_y = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # 3) Take the absolute value of the x and y gradients
    magnitude = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    grad_dir = np.arctan2(sobel_y, sobel_x)
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(grad_dir, dtype=np.uint8)
    thresh_min, thresh_max = thresh
    binary_output[(grad_dir >= thresh_min) & (grad_dir <= thresh_max)] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

def hls_select(img, thresh_h=(0, 180), thresh_l=(0, 255), thresh_s=(0, 255)):
    # 1) Convert to HLS color space
    h, l, s = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HLS))
    # 2) Apply a threshold to the S channel
    binary_output = np.zeros_like(s, dtype=np.uint8)
    thresh_min, thresh_max = thresh_s

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # white color section in HSV space:[(h_min, h_mAX), (s_min, s_max), (v_min, v_max)]
    hsv_write_threshold = [(0, 180), (0, 30), (120, 255)]
    # select white region
    white_thresholds = ((hsv[:,:,0] >= hsv_write_threshold[0][0]) & (hsv[:,:,0] <= hsv_write_threshold[0][1])) & \
            ((hsv[:,:,1] >= hsv_write_threshold[1][0]) & (hsv[:,:,1] <= hsv_write_threshold[1][1])) & \
            ((hsv[:,:,2] >= hsv_write_threshold[2][0]) & (hsv[:,:,2] <= hsv_write_threshold[2][1]))

    # yellow color section in HSV space
    hsv_yellow_threshold = [(90, 100), (43, 255), (46, 255)]
    yellow_thresholds = ((hsv[:,:,0] >= hsv_yellow_threshold[0][0]) & (hsv[:,:,0] <= hsv_yellow_threshold[0][1])) & \
            ((hsv[:,:,1] >= hsv_yellow_threshold[1][0]) & (hsv[:,:,1] <= hsv_yellow_threshold[1][1])) & \
            ((hsv[:,:,2] >= hsv_yellow_threshold[2][0]) & (hsv[:,:,2] <= hsv_yellow_threshold[2][1]))

    # binary_output[(s >= thresh_min) & (s <= thresh_max) & (white_thresholds | yellow_thresholds)] = 1
    binary_output[(s >= thresh_min) & (s <= thresh_max)] = 1

    # 3) Return a binary image of threshold result
    return binary_output

 
param_dict = {}
param_dict['gradx'] = [int, (0, 255), 1, [10, 100]]
param_dict['grady'] = [int, (0, 255), 1, [10, 100]]
param_dict['mag'] = [int, (0, 255), 1, [30, 100]]
param_dict['dir'] = [float, (0, 314), 0.01, [0.7, 1.2]]
# white threshold in hsv color space
# param_dict['hls_h'] = [int, (0, 180), 1,[0, 180]]
# param_dict['hls_l'] = [int, (0, 255), 1,[0, 30]]
# param_dict['hls_s'] = [int, (0, 255), 1,[120, 255]]

# yellow threshold in hsv color space
param_dict['hls_h'] = [int, (0, 180), 1,[90, 100]] # h
param_dict['hls_l'] = [int, (0, 255), 1,[200, 255]] # s
param_dict['hls_s'] = [int, (0, 255), 1,[60, 255]] # v

# white threshold in hls color space
# param_dict['hls_h'] = [int, (0, 180), 1,[0, 180]]
# param_dict['hls_l'] = [int, (0, 255), 1,[105, 255]]
# param_dict['hls_s'] = [int, (0, 255), 1,[0, 255]]

# yellow threshold in hls color space
# param_dict['hls_h'] = [int, (0, 180), 1,[0, 180]]
# param_dict['hls_l'] = [int, (0, 255), 1,[105, 255]]
# param_dict['hls_s'] = [int, (0, 255), 1,[0, 255]]

def threshold_image(image, debug=False):
    global should_exit
    def do_nothing(x):
        pass

    def on_mouse_event(event,x,y,flags,param):
       if event==cv2.EVENT_FLAG_LBUTTON:
           print(( 'press:', x, y))

    image = cv2.GaussianBlur(image, (5, 5), 0)

    # type, show_range, scale, default_value, 
    if debug:
        default_value = {}
        for name, param in list(param_dict.items()):
            p_type = param[0]
            p_range = param[1]
            p_scale = param[2]
            p_default = param[3]

            cv2.namedWindow(name)
            cv2.createTrackbar('Min',name,p_range[0],p_range[1],do_nothing)
            cv2.createTrackbar('Max',name,p_range[0],p_range[1],do_nothing)

            cv2.setTrackbarPos('Min',name, int(p_default[0]/p_scale))
            cv2.setTrackbarPos('Max',name, int(p_default[1]/p_scale))

            default_value[name] = (int(p_default[0]/p_scale), int(p_default[1]/p_scale))

    while True:
        ksize = 3 # Choose a larger odd number to smooth gradient measurements
        if debug:
            new_value = {}
            for name, param in list(param_dict.items()):
                min_val = cv2.getTrackbarPos('Min',name)
                max_val = cv2.getTrackbarPos('Max',name)
                param[3][0] = min_val * param[2]
                param[3][1] = max_val * param[2]
                new_value[name] = (min_val, max_val)

            if len([1 for name in list(new_value.keys()) if default_value[name]!=new_value[name]]) > 0:
                print (new_value)
                default_value = {}
                default_value.update(new_value)

        pd = param_dict

        # Apply each of the thresholding functions
        gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(pd['gradx'][3][0], pd['gradx'][3][1]))
        grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(pd['grady'][3][0], pd['grady'][3][1]))
        mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(pd['mag'][3][0], pd['mag'][3][1]))
        dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(pd['dir'][3][0], pd['dir'][3][1]))
        hls_binary = hls_select(image, thresh_h=(pd['hls_h'][3][0], pd['hls_h'][3][1]), thresh_l=(pd['hls_l'][3][0], pd['hls_l'][3][1]), thresh_s=(pd['hls_s'][3][0], pd['hls_s'][3][1]))
        combined = np.zeros_like(dir_binary)
        combined[(((gradx == 1) & (grady == 1) & (hls_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1))  | (hls_binary == 1))] = 1
        # combined[(((gradx == 1) & (grady == 1)) )] = 1

        if debug:
            cv2.imshow('image', image)
            cv2.setMouseCallback('image', on_mouse_event)
            cv2.imshow('gradx', gradx.astype(np.float32))
            cv2.imshow('grady', grady.astype(np.float32))
            cv2.imshow('mag', mag_binary.astype(np.float32))
            cv2.imshow('dir', dir_binary.astype(np.float32))
            cv2.imshow('hls_s', hls_binary.astype(np.float32))
            cv2.imshow('combined', combined.astype(np.float32))
            key = cv2.waitKey(100) & 0xff
            if key in [ord('q'), ord('Q'), 23]:
                should_exit = True
                break
            if key in [ord(' ')]:
                break
            if key in [ord('s'), ord('S')]:
                cv2.imwrite('./output_images/test_gradx.jpg', gradx * 255)
                cv2.imwrite('./output_images/test_grady.jpg', grady * 255)
                cv2.imwrite('./output_images/test_mag.jpg', mag_binary * 255)
                cv2.imwrite('./output_images/test_dir.jpg', dir_binary * 255)
                cv2.imwrite('./output_images/test_hls.jpg', hls_binary * 255)
                cv2.imwrite('./output_images/test_combined.jpg', combined * 255)
        else:
            break
    return combined


lane_line_points = ((236, 720), (617, 438), (666, 438), (1099, 720))
# project_video.ma4
lane_line_points = ((303, 689), (584, 467), (716, 467), (1098, 689))

def undisort_image(image, mtx, dist):
    undis_image = cv2.undistort(image, mtx, dist, None, mtx)
    return undis_image

def get_transform_mat(offset=200, shape=(1280, 720)):
    # points in un-distort image:
    # down_left, up_left, up_right, down_right
    w, h = shape

    src_points = np.array(lane_line_points, dtype=np.float32)

    x_left, x_right = 200, 800

    # get tranform matrix from un-distort view to top-down view.
    dst_points = np.array([(x_left, h), (x_left, 200), (x_right, 200), (x_right, h)], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    return M

def transform_image(image, M, shape=(1280, 720)):
    w, h = shape
    top_down = cv2.warpPerspective(image, M, (w, h))
    return top_down

def finding_line(binary):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

def finding_line_with_preframe_line(binary, pre_frame_line):
    left_fit, right_fit = pre_frame_line
    mask = np.zeros_like(binary, dtype=np.uint8)
    y = np.linspace(0, 719, num=72).astype(np.int32)
    leftx = np.int0(left_fit[0]*y**2 + left_fit[1]*y + left_fit[2])
    for pi in range(len(y) - 1):
        cv2.line(mask, (leftx[pi+1], y[pi+1]), (leftx[pi], y[pi]), (255,), 20)

    warped = binary & mask
    l_y, l_x = np.where(warped > 0)
    new_left_fit = np.polyfit(l_y, l_x, 2)


    mask = np.zeros_like(binary, dtype=np.uint8)
    rightx = np.int0(right_fit[0]*y**2 + right_fit[1]*y + right_fit[2])
    for pi in range(len(y) - 1):
        cv2.line(mask, (rightx[pi+1], y[pi+1]), (rightx[pi], y[pi]), (255,), 20)

    warped = binary & mask
    r_y, r_x = np.where(warped > 0)
    new_right_fit = np.polyfit(r_y, r_x, 2)
    # new_right_fit = new_left_fit.copy()
    # new_right_fit[2] += 620

    y_test_list = [0]
    offset = 0
    for y_test in y_test_list:
        a, b, c = left_fit[:3]
        x = a * (y_test ** 2) + b * y_test + c

        a, b, c = new_left_fit[:3]
        new_x = a * (y_test ** 2) + b * y_test + c

        offset += abs(new_x - x)
    if offset > 100:
        new_left_fit = left_fit

    y_test_list = [0]
    offset = 0
    for y_test in y_test_list:
        a, b, c = right_fit[:3]
        x = a * (y_test ** 2) + b * y_test + c

        a, b, c = new_right_fit[:3]
        new_x = a * (y_test ** 2) + b * y_test + c

        offset += abs(new_x - x)
    if offset > 100:
        new_right_fit = right_fit

    empty = np.zeros_like(binary)
    empty = np.dstack([empty, empty, empty])
    leftx = np.int0(new_left_fit[0]*y**2 + new_left_fit[1]*y + new_left_fit[2])
    rightx = np.int0(new_right_fit[0]*y**2 + new_right_fit[1]*y + new_right_fit[2])
    for pi in range(len(y) - 1):
        cv2.line(empty, (leftx[pi], y[pi]), (rightx[pi], y[pi]), (0, 255, 0), 10)
    for pi in range(len(y) - 1):
        cv2.line(empty, (leftx[pi+1], y[pi+1]), (leftx[pi], y[pi]), (0, 0, 255), 10)
        cv2.line(empty, (rightx[pi+1], y[pi+1]), (rightx[pi], y[pi]), (0, 0, 255), 10)

    return empty, [new_left_fit, new_right_fit]

def finding_line_slide_window(binary, debug=False, output_line=None):
    warped = binary
    # window settings
    window_width = 60 
    window_height = 180 # Break image into 9 vertical layers since image height is 720
    margin = 100 # How much to slide left and right for searching

    def window_mask(width, height, img_ref, center,level):
        output = np.zeros_like(img_ref)
        output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
        return output

    def find_window_centroids(image, window_width, window_height, margin):
        
        window_centroids = [] # Store the (left,right) window centroid positions per level
        window = np.ones(window_width) # Create our window template that we will use for convolutions
        
        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template 
        
        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
        l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
        r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
        r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
        # print r_center, -window_width/2+int(image.shape[1]/2)
        
        # Add what we found for the first layer
        window_centroids.append((l_center,r_center))
        
        # Go through each layer looking for max pixel locations
        # for level in range(1,(int)(image.shape[0]/window_height)):
        for level in range(1,(int)(image.shape[0]/window_height)):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
            offset = window_width/2
            l_min_index = int(max(l_center+offset-margin,0))
            l_max_index = int(min(l_center+offset+margin,image.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center+offset-margin,0))
            r_max_index = int(min(r_center+offset+margin,image.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
            # print r_center, -window_width/2+int(image.shape[1]/2)
            # Add what we found for that layer
            window_centroids.append((l_center,r_center))

        return window_centroids

    window_centroids = find_window_centroids(warped, window_width, window_height, margin)

    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)

        # Go through each level and draw the windows    
        for level in range(0,len(window_centroids)):

            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
            r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
            # Add graphic points from window mask here to total pixels found 
            l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        # cv2.imshow('r_points', r_points)
        # cv2.imshow('l_points', l_points)
        ##################
        # fit right pixes again
        r_mask = np.array(r_points,np.uint8) 
        r_warp = warped.copy()
        r_warp[np.where(r_mask==0)] = 0
        # cv2.imshow('r_warp', r_warp * 255)
        # cv2.imshow('warp', warped.astype(np.float32))
        r_y, r_x = np.where(r_warp > 0)
        ##################
        ##################
        # fit left pixes again
        l_mask = np.array(l_points,np.uint8)
        l_warp = warped.copy()
        l_warp[np.where(l_mask==0)] = 0
        # cv2.imshow('l_warp', l_warp * 255)
        l_y, l_x = np.where(l_warp > 0)
        ##################
        # draw lane lines
        zero_channel = np.zeros_like(warped) # create a zero color channel
        left_fit = np.polyfit(l_y, l_x, 2)
        # print 'left fit:', left_fit
        left_fitx = left_fit[0]*l_y**2 + left_fit[1]*l_y + left_fit[2]
        # right_fit = np.polyfit(r_y, r_x, 2)
        right_fit = left_fit.copy()
        right_fit[2] += 620
        
        right_fitx = right_fit[0]*r_y**2 + right_fit[1]*r_y + right_fit[2]
        y = np.linspace(0, 719, num=72).astype(np.int32)
        leftx = np.int0(left_fit[0]*y**2 + left_fit[1]*y + left_fit[2])
        rightx = np.int0(right_fit[0]*y**2 + right_fit[1]*y + right_fit[2])
        empty = np.zeros_like(r_warp)
        empty = np.dstack([empty, empty, empty])
        for pi in range(len(y) - 1):
            cv2.line(empty, (leftx[pi], y[pi]), (rightx[pi], y[pi]), (0, 255, 0), 10)
        for pi in range(len(y) - 1):
            cv2.line(empty, (leftx[pi+1], y[pi+1]), (leftx[pi], y[pi]), (0, 0, 255), 10)
            cv2.line(empty, (rightx[pi+1], y[pi+1]), (rightx[pi], y[pi]), (0, 0, 255), 10)
        warpage= np.dstack((warped, warped, warped))*255 # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 0.5, empty, 0.5, 0.0) # overlay the orignal road image with window results
        # cv2.imwrite("output_images/test1_color_fit.jpg", output)
        # cv2.imshow('output_undistort', empty)
        ##################
        # output = empty
        if type(output_line) is list:
            output_line.append(left_fit)
            output_line.append(right_fit)
     
    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((warped,warped,warped)),np.uint8)

    if debug:
        # Display the final results
        plt.imshow(output)
        plt.title('window fitting results')
        plt.show()
    return output

def calc_line_curvature(line, y=0):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30.0/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    a, b, c = line[:3]
    # unit from pixel to meters
    a = a / (ym_per_pix ** 2) * xm_per_pix
    b = b / ym_per_pix * xm_per_pix
    c = c * xm_per_pix
    y = y * ym_per_pix
    r_curv = pow((1 + pow(2 * a * y + b, 2)), 3/2.0) / np.abs(a * 2)
    return r_curv

def calc_left_fo_center(line1, line2, y=0):
    a, b, c = line1[:3]
    x1 = a * (y**2) + b * y + c

    a, b, c = line2[:3]
    x2 = a * (y**2) + b * y + c

    ym_per_pix = 30.0/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    return abs(((x2 + x1) / 2.0 - 640) * xm_per_pix)

def find_lane_line_image(image, pre_frame_result=None):
    global should_pause
    fn = './calibration_data.pickle'
    with open('./calibration_data.pickle', 'r') as fd:
        calibration_data = pickle.load(fd)

    mtx, dist = calibration_data['mtx'], calibration_data['dist']
    image = undisort_image(image, mtx, dist)
    origin = image.copy()

    h = 720
    # src_points = lane_line_points
    # for i in range(len(src_points)):
    #     cv2.line(origin, src_points[i-1], src_points[i], (255, 0, 0), 4)
    # cv2.imshow('origin', origin)

    transform_mat = get_transform_mat()

    image = transform_image(image, transform_mat)

    threshold = threshold_image(image, debug=False)

    # cv2.imwrite("./output_images/test1_warped_straight_lines.jpg", np.hstack((origin, image)))
    # threshold = transform_image(threshold, transform_mat)


    # cv2.imshow('threshold', threshold.astype(np.float32))
    output_lines = []
    
    threshold = threshold * 255
    threshold = threshold.astype(np.uint8)
    if pre_frame_result is None or len(pre_frame_result) < 2:
        line_image = finding_line_slide_window(threshold, output_line = output_lines)
    else:
        line_image, output_lines = finding_line_with_preframe_line(threshold, pre_frame_result)
    
    left_fit, right_fit = output_lines

    s1 = 'Radius of Curvature = %5.1f (m)' % calc_line_curvature(left_fit, 720)
    s2 = 'Vehicle is  %4.2f m left of center' % calc_left_fo_center(left_fit, right_fit, 720)
    
    line_image = transform_image(line_image, np.linalg.inv(transform_mat)).astype(np.uint8)
    line_image = cv2.addWeighted(origin, 0.8, line_image, 0.2, 0)
    cv2.putText(line_image, s1, (10, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255), 1, False)
    cv2.putText(line_image, s2, (10, 120), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255), 1, False)
    # cv2.imwrite("output_images/test1_output.jpg", line_image)
    cv2.imshow('lane_line', line_image.astype(np.uint8))
    # cv2.imwrite('./output_images/test_lane_line.jpg', line_image.astype(np.uint8))
    # key = cv2.waitKey(10) & 0xff
    # if key in [ord(' ')]:
        # should_pause = not should_pause
    return line_image.astype(np.uint8), output_lines
    
def process_image(arg):
    fn_list = []
    if os.path.isfile(arg.input):
        fn_list = [arg.input]
    elif os.path.isdir(arg.input):
        fn_list = os.listdir(arg.input)
        ext_list = ['.jpg', '.bmp', '.png', '.jpeg']
        fn_list = [os.path.join(arg.input, fn) for fn in fn_list if os.path.splitext(fn)[1] in ext_list]
    else:
        fn_list = []

    if len(fn_list) < 1:
        print ("No image file found.")

    for fn in fn_list:
        print( ("process image filename :", fn))
        image = cv2.imread(fn)
        find_lane_line_image(image)
        if should_exit:
            break

    pass

def process_video(arg):
    global should_pause, should_exit
    def on_mouse_event(event,x,y,flags,param):
       if event==cv2.EVENT_FLAG_LBUTTON:
           print(( 'press:', x, y))
    print (("process_video...", arg.input))
    if os.path.isfile(arg.input):
        video_fn = arg.input
    else:
        return None
    cap = cv2.VideoCapture(video_fn)

    if arg.output:
        output_fn = arg.output
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)  
        size = (int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),   
                int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))  
        out = cv2.VideoWriter(output_fn, cv2.cv.CV_FOURCC('D', 'I', 'V', 'X'), fps, size, 1) 
    else:
        out = None
    cap_index = 0
    # should_pause = True
    result = None
    while True:
        # get a frame
        ret, frame = cap.read()
        cap_index += 1
        if ret:
            if cap_index < 0:
                continue
            image, result = find_lane_line_image(frame, result)
            # print "image index[from 1]", cap_index
            if out:
                out.write(image)
            

            while True:         

                # cv2.imshow("test", frame)
                # cv2.setMouseCallback('test', on_mouse_event)
                cv2.imshow('lane_line', image)
                key = cv2.waitKey(10) & 0xff
                if key in [ord(' ')]:
                    should_pause = not should_pause
                elif key in [ord('q'), 23, ord('Q')]:
                    should_exit = True
                    break
                elif key in [ord('n'), ord('N')]:
                    should_pause = True
                    break
                if not should_pause:
                    break
                if should_exit:
                    break
                time.sleep(0.1)
        else:
            break
        if should_exit:
            break
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="find lane line in image or videos")
    subparsers = parser.add_subparsers(help='commands')

    image_parse = subparsers.add_parser('image', help='calibrate a camera with chessboard pictures')
    image_parse.set_defaults(func=process_image)
    image_parse.add_argument("input", action='store',help='image file or directory.')
    image_parse.add_argument("--output", action='store',help='save result image to another file')

    video_parse = subparsers.add_parser('video', help='calibrate a camera with chessboard pictures')
    video_parse.set_defaults(func=process_video)
    
    video_parse.add_argument("input", action='store',help='*.mp4 file or directory.')
    video_parse.add_argument("--output", action='store',help='save result video to another file')
     
    args = parser.parse_args(sys.argv[1:])
    args.func(args)
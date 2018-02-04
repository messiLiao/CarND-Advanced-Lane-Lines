import os
import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt
import argparse

try:
    import pickle
except:
    import cPickle as pickle

should_exit = False

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
    thresh_min, thresh_max = 170, 255

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
param_dict['gradx'] = [int, (0, 255), 1, [20, 100]]
param_dict['grady'] = [int, (0, 255), 1, [20, 100]]
param_dict['mag'] = [int, (0, 255), 1, [30, 100]]
param_dict['dir'] = [float, (0, 314), 0.01, [0.7, 1.2]]
# white threshold in hsv color space
# param_dict['hls_h'] = [int, (0, 180), 1,[0, 180]]
# param_dict['hls_l'] = [int, (0, 255), 1,[0, 30]]
# param_dict['hls_s'] = [int, (0, 255), 1,[120, 255]]

# yellow threshold in hsv color space
param_dict['hls_h'] = [int, (0, 180), 1,[90, 100]] # h
param_dict['hls_l'] = [int, (0, 255), 1,[43, 255]] # s
param_dict['hls_s'] = [int, (0, 255), 1,[46, 255]] # v

# white threshold in hls color space
# param_dict['hls_h'] = [int, (0, 180), 1,[0, 180]]
# param_dict['hls_l'] = [int, (0, 255), 1,[105, 255]]
# param_dict['hls_s'] = [int, (0, 255), 1,[0, 255]]

# yellow threshold in hls color space
# param_dict['hls_h'] = [int, (0, 180), 1,[0, 180]]
# param_dict['hls_l'] = [int, (0, 255), 1,[105, 255]]
# param_dict['hls_s'] = [int, (0, 255), 1,[0, 255]]

def threshold_image(image):
    global should_exit
    def do_nothing(x):
        pass

    image = cv2.GaussianBlur(image, (5, 5), 0)

    # type, show_range, scale, default_value, 
    default_value = {}
    for name, param in param_dict.items():
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
        new_value = {}
        for name, param in param_dict.items():
            min_val = cv2.getTrackbarPos('Min',name)
            max_val = cv2.getTrackbarPos('Max',name)
            param[3][0] = min_val * param[2]
            param[3][1] = max_val * param[2]
            new_value[name] = (min_val, max_val)

        if len([1 for name in new_value.keys() if default_value[name]!=new_value[name]]):
            print new_value
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
        combined[(((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))  | (hls_binary == 1))] = 1
        # combined[(hls_binary == 1)] = 1

        cv2.imshow('image', image)
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
    return combined

def undisort_image(image, mtx, dist):
    undis_image = cv2.undistort(image, mtx, dist, None, mtx)
    return undis_image

def find_lane_line_image(image):
    fn = './calibration_data.pickle'
    with open('./calibration_data.pickle', 'r') as fd:
        calibration_data = pickle.load(fd)

    mtx, dist = calibration_data['mtx'], calibration_data['dist']
    image = undisort_image(image, mtx, dist)

    image = threshold_image(image)


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
        print "No image file found."

    for fn in fn_list:
        print "process image filename :", fn
        image = cv2.imread(fn)
        find_lane_line_image(image)
        if should_exit:
            break

    pass

def process_video(arg):
    print "process_video...", arg.input
    if os.path.isfile(arg.input):
        video_fn = arg.input
    else:
        return None
    cap = cv2.VideoCapture(video_fn)
    while True:
        # get a frame
        ret, frame = cap.read()
        if ret:
            find_lane_line_image(frame)
        if should_exit:
            break
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="find lane line in image or videos")
    subparsers = parser.add_subparsers(help='commands')

    image_parse = subparsers.add_parser('image', help='calibrate a camera with chessboard pictures')
    image_parse.set_defaults(func=process_image)

    video_parse = subparsers.add_parser('video', help='calibrate a camera with chessboard pictures')
    video_parse.set_defaults(func=process_video)
    
    parser.add_argument("input", action='store',help='image file or *.mp4 file or directory.')
     
    args = parser.parse_args(sys.argv[1:])
    args.func(args)
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

def calibration_images(arg):
    fn_list = [os.path.join(arg.dirname, fn) for fn in os.listdir(arg.dirname)]
    imgpoints = []
    objpoints = []
    objp = np.zeros((6*9,3), np.float32)  
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    for fn in fn_list:
        image = cv2.imread(fn)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(image, (9, 6), None)
        if ret:
            imgpoints.append(corners)
            r = cv2.drawChessboardCorners(image, (9, 6), corners, ret)
            objpoints.append(objp)
            print "found corners..", fn
            if arg.output:
                f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
                f.tight_layout()
                ax1.imshow(img)
                ax1.set_title('Original Image', fontsize=50)
                ax2.imshow(top_down)
                ax2.set_title('Undistorted and Warped Image', fontsize=50)
                plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

        else:
            print 'corners not found in ', fn

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (1280, 720), None, None)
    calibration_data = {}
    calibration_data['ret'] = ret
    calibration_data['mtx'] = mtx
    calibration_data['dist'] = dist
    calibration_data['rvecs'] = rvecs
    calibration_data['tvecs'] = tvecs
    with open('./calibration_data.pickle', 'w') as fd:
        pickle.dump(calibration_data, fd)
        print "calibration data save to \"calibration_data.pickle\" successfully."

def distort_image(arg):
    fn = arg.image_name
    image = cv2.imread(fn)
    with open('./calibration_data.pickle', 'r') as fd:
        calibration_data = pickle.load(fd)

    mtx, dist = calibration_data['mtx'], calibration_data['dist']
    undis_image = cv2.undistort(image, mtx, dist, None, mtx)
    if arg.show:
        cv2.imshow('undist', undis_image)
        cv2.imshow('image', image)
        cv2.waitKey(0)
    if arg.save:
        undist_fn = arg.save
        cv2.imwrite(undist_fn, undis_image)
        print "save to file :", undist_fn
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='commands')

    cali_parse = subparsers.add_parser('calibrate', help='calibrate a camera with chessboard pictures')
    cali_parse.add_argument("dirname", action='store',help='The directory which contains chessboard pictures')
    cali_parse.add_argument("--output", action='store', help='Save un-distort image to a directory')
    cali_parse.set_defaults(func=calibration_images)
    dist_parse = subparsers.add_parser('distort', help='distort a image')
    dist_parse.add_argument("image_name", action='store',help='The image filename')
    dist_parse.add_argument("--save", action='store',help='save the undistort image to file')
    dist_parse.add_argument("--show", action='store_true',help='show image and undistort image')
    dist_parse.set_defaults(func=distort_image)
     
    args = parser.parse_args(sys.argv[1:])
    args.func(args)

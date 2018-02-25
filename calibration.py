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

def calibrate_main(arg):
    if os.path.isdir(arg.dirname):
        calibration_images(arg)
    else:
        calibration_video(arg)

def transform_main(arg):
    fn = arg.filename
    f, ext = os.path.splitext(fn)
    if ext in ['.mp4', '.avi', '.mpg']:
        transform_video(fn)
    elif ext in ['.jpg', '.jpeg', '.bmp', '.png']:
        transform_image(fn)
    else:
        pass
        pass

def transform_video(fn):
    point_list = [(303, 689), (584, 467), (716, 467), (1098, 689)]
    def on_mouse_event(event,x,y,flags,param):
       if event==cv2.EVENT_FLAG_LBUTTON:
           print(( 'press:', x, y))
           add_cross_point((x, y))
           print point_list
    def add_cross_point(point):
        index = -1
        distance = 9e9
        x, y = point
        for i, (_x, _y) in enumerate(point_list):
            dis = (x - _x) ** 2 + (y - _y) ** 2
            if dis < distance:
                index = i
                distance = dis
        if index >= 0:
            point_list[index] = point
    cap = cv2.VideoCapture(fn)
    pause = False
    exit = False
    with open('./calibration_data.pickle', 'r') as fd:
        calibration_data = pickle.load(fd)

    mtx, dist = calibration_data['mtx'], calibration_data['dist']
    skip_index = 0
    image_index = 0
    if cap:
        while not exit:
            ret, frame = cap.read()
            image_index += 1
            if image_index < skip_index:
                continue

            undis_image = cv2.undistort(frame, mtx, dist, None, mtx)
            if ret:
                while True:
                    # cv2.imshow('frame', frame)
                    line_image = undis_image.copy()
                    cv2.line(line_image, point_list[0], point_list[1], (0, 255, 200), 10)
                    cv2.line(line_image, point_list[2], point_list[3], (0, 255, 200), 10)
                    # cv2.imshow('undistort', line_image)

                    src_points = np.array(point_list, dtype=np.float32)
                    h, w = line_image.shape[:2]
                    x_left, x_right = 200, 1080
                    dst_points = np.array([(x_left, h), (x_left, 100), (x_right, 100), (x_right, h)], dtype=np.float32)
                    M = cv2.getPerspectiveTransform(src_points, dst_points)
                    top_down = cv2.warpPerspective(undis_image, M, (w, h))
                    # cv2.imshow('Perspective', top_down)

                    cv2.setMouseCallback('undistort', on_mouse_event)
                    key = cv2.waitKey(50) & 0xff
                    if key in [ord(' ')]:
                        pause = not pause
                    elif key in [ord('q'), 23]:
                        exit = True
                        break
                        pass
                    if not pause:
                        break
            else:
                break
    pass

def transform_image(fn):
    pass

def calibration_video(arg):
    pass

def calibration_images(arg):
    fn_list = [os.path.join(arg.dirname, fn) for fn in os.listdir(arg.dirname)]
    imgpoints = []
    objpoints = []
    offset = 100
    objp = np.zeros((6*9,3), np.float32)  
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    found_fn_dict = {}
    # step1: find corners in images
    for fn in fn_list:
        image = cv2.imread(fn)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(image, (9, 6), None)
        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)
            print "found corners..", fn
            found_fn_dict[fn] = corners

        else:
            print 'corners not found in ', fn

    # step2: calerate camera by corners
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (1280, 720), None, None)
    print mtx.shape, dist.shape, len(rvecs), len(tvecs)

    #step3: undistort image and corners.
    for fn, corners in found_fn_dict.items(): 
        image = cv2.imread(fn)
        cv2.drawChessboardCorners(image, (9, 6), corners, True)
        h, w = image.shape[:2]

        # un-distort images
        undis_image = cv2.undistort(image, mtx, dist, None, mtx)

        # un-distort corners
        src_corners = corners[[0, 8, -9, -1]]
        undis_src_corners = cv2.undistortPoints(src_corners, mtx, dist, None, mtx)

        # get tranform matrix from un-distort view to top-down view.
        dst_corners = np.array([(offset, offset), (w - offset, offset), (offset, h - offset), (w - offset, h - offset)], dtype=np.float32)
        print fn, undis_src_corners, dst_corners
        break
        M = cv2.getPerspectiveTransform(undis_src_corners, dst_corners)

        # transform un-distort image to top-down view.
        top_down = cv2.warpPerspective(undis_image, M, (w, h))
        if arg.show or os.path.isdir(arg.output):
            f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
            f.tight_layout()
            ax1.imshow(image)
            ax1.set_title('Original Image', fontsize=30)
            ax2.imshow(undis_image)
            ax2.set_title('Undistorted Image', fontsize=30)
            ax3.imshow(top_down)
            ax3.set_title('Undistorted and Warped Image', fontsize=30)
            plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

            # save image to directory if needed
            if os.path.isdir(arg.output):
                base, f = os.path.split(fn)
                new_fn = os.path.join(arg.output, f)
                print new_fn
                plt.savefig(new_fn)
            # display image if needed
            if arg.show:
                plt.show()
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
    parser = argparse.ArgumentParser(description="camera calerate tool")
    subparsers = parser.add_subparsers(help='commands')

    cali_parse = subparsers.add_parser('calibrate', help='calibrate a camera with chessboard pictures')
    cali_parse.add_argument("dirname", action='store',help='The directory which contains chessboard pictures')
    cali_parse.add_argument("--output", action='store', help='Save un-distort image to a directory')
    cali_parse.add_argument("--show", action='store_true', help='Show un-distort image')
    cali_parse.set_defaults(func=calibrate_main)

    dist_parse = subparsers.add_parser('distort', help='distort a image')
    dist_parse.add_argument("image_name", action='store',help='The image filename')
    dist_parse.add_argument("--save", action='store',help='save the undistort image to file')
    dist_parse.add_argument("--show", action='store_true',help='show image and undistort image')
    dist_parse.set_defaults(func=distort_image)

    trans_parse = subparsers.add_parser('transform', help='calibrate a camera with chessboard pictures')
    trans_parse.add_argument("filename", action='store',help='The directory which contains chessboard pictures')
    trans_parse.add_argument("--output", action='store', help='Save un-distort image to a directory')
    trans_parse.add_argument("--show", action='store_true', help='Show un-distort image')
    trans_parse.set_defaults(func=transform_main)
     
    args = parser.parse_args(sys.argv[1:])
    args.func(args)

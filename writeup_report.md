## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration17.jpg "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image2_undistort]: ./output_images/test1_undistort.jpg "Road Undistort"
[image2_combined]: ./output_images/test_combined.jpg "Combined binary image"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image2_warped]: ./output_images/test1_warped_straight_lines.jpg "Warp Example"
[image2_color_fit]: ./output_images/test1_color_fit.jpg "Fit Visual"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[image2_output]: ./output_images/test1_output.jpg "Output"
[video1]: ./project_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained  in lines 13 through 88 of the file called `calibration.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

distortion correction result image:
![alt text][image2_undistort]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 128 through 209 in `findLaneLine.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image2_combined]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `get_transform_mat()`, which appears in lines 220 through 232 in the file `findLaneline.py`.  The `get_transform_mat()` function takes as source (`src_points`) and destination (`dst_points`) points, and retururn perspective transform matrix.  I chose the hardcode the source and destination points in the following manner:

```python
lane_line_points = ((303, 689), (584, 467), (716, 467), (1098, 689))
src_points = np.array(lane_line_points, dtype=np.float32)

x_left, x_right = 200, 800

# get tranform matrix from un-distort view to top-down view.
dst_points = np.array([(x_left, h), (x_left, 200), (x_right, 200), (x_right, h)], dtype=np.float32)
M = cv2.getPerspectiveTransform(src_points, dst_points)
```

This resulted in the following source and destination points:
| Source        | Destination   | 
|:-------------:|:-------------:| 
| 303, 689      | 200, 720        | 
| 584, 467      | 200, 200      |
| 716, 467     | 800, 200      |
| 1098, 689      | 800, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image2_warped]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image2_color_fit]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 505 through 516 in my code in `findLneLine.py` in the function `calc_line_curvature()`. The unit of x, and y in the fitted function is pixel, I should change the unit to meter. Every coefficient should multiply parameters.
```python
ym_per_pix = 30.0/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
a, b, c = line[:3]
# unit from pixel to meters
a = a / (ym_per_pix ** 2) * xm_per_pix
b = b / ym_per_pix * xm_per_pix
c = c * xm_per_pix
y = y * ym_per_pix
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 529 through 580 in my code in `findLaneLine.py` in the function `find_lane_line_image()`.  Here is an example of my result on a test image:

![alt text][image2_output]

---

### Pipeline (video)
#### 1. Solution
After the previous analysis and testing, My last solution described as follow steps. I implemented these steps in lines 529 through 580 in my code in `findLaneLine.py` in the function `find_lane_line_image()`.
Step 1: Undisorting image and get the top-view image. Lines 530 through 547 in my code in `findLaneLine.py`
Step 2: I used a combination of color and gradient thresholds to generate a binary image. Lines in the 549.
Step 1: Find the lane line in the global image in the first frame. Lines 558 through 561 in my code in `findLaneLine.py`
Step 2: If found, then find the lane line in the previous lane line mask in the current frame, if not the repeat step 1. Lines in the 563 in my code in `findLaneLine.py`
Step 3: Compare the lane line in the previous frame and the current. If not similar, the use a previous lane line as current. Lines 312 through 372 in my code in `findLaneLine.py` in the function `finding_line_with_preframe_line()`.
Step 4: Until end.
#### 2. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_output.avi)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

when I process a frame in a video, I didn't use information previous frame. Avalible information which can be radius of curvature of the lane, and lane line distance and so on. Hough line method alse can use here: fx = ax^3 + bx^2 + c;

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

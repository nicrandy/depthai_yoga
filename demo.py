#!/usr/bin/env python3

from BlazeposeRenderer import BlazeposeRenderer
import argparse
import cv2
from sklearn import svm
import pickle
import time
import os
from random import sample
import array
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--edge', action="store_true",
                    help="Use Edge mode (postprocessing runs on the device)")
parser.add_argument('-i', '--input', type=str, default="rgb", 
                    help="'rgb' or 'rgb_laconic' or path to video/image file to use as input (default=%(default)s)")
parser.add_argument("--pd_m", type=str,
                    help="Path to an .blob file for pose detection model")
parser.add_argument("--lm_m", type=str,
                    help="Landmark model ('full' or 'lite' or '831') or path to an .blob file (default=%(default)s)")
parser.add_argument('-c', '--crop', action="store_true", 
                    help="Center crop frames to a square shape before feeding pose detection model")
parser.add_argument('--no_smoothing', action="store_true", 
                    help="Disable smoothing filter")
parser.add_argument('--filter_window_size', type=int, default=5,
                    help="Smoothing filter window size. Higher value adds to lag and to stability (default=%(default)i)")                    
parser.add_argument('--filter_velocity_scale', type=float, default=10,
                    help="Smoothing filter velocity scale. Lower value adds to lag and to stability (default=%(default)s)")                    
parser.add_argument('-f', '--internal_fps', type=int, 
                    help="Fps of internal color camera. Too high value lower NN fps (default= depends on the model)")                    
parser.add_argument('--internal_frame_height', type=int, default=640,                                                                                    
                    help="Internal color camera frame height in pixels (default=%(default)i)")                    
parser.add_argument('-s', '--stats', action="store_true", 
                    help="Print some statistics at exit")
parser.add_argument('-t', '--trace', action="store_true", 
                    help="Print some debug messages")
parser.add_argument('--force_detection', action="store_true", 
                    help="Force person detection on every frame (never use landmarks from previous frame to determine ROI)")

parser.add_argument('-3', '--show_3d', action="store_true", 
                    help="Display skeleton in 3d in a separate window (valid only for full body landmark model)")
parser.add_argument("-o","--output",
                    help="Path to output video file")
 

args = parser.parse_args()

from BlazeposeDepthaiEdge import BlazeposeDepthai

pose = BlazeposeDepthai(input_src=args.input, 
            pd_model=args.pd_m,
            lm_model=args.lm_m,
            smoothing=not args.no_smoothing,
            filter_window_size=args.filter_window_size,
            filter_velocity_scale=args.filter_velocity_scale,                
            crop=args.crop,
            internal_fps=args.internal_fps,
            internal_frame_height=args.internal_frame_height,
            force_detection=args.force_detection,
            stats=args.stats,
            trace=args.trace)   

renderer = BlazeposeRenderer(
                pose, 
                show_3d=args.show_3d, 
                output=args.output)
############### New stuff ###############################
#global variables
current_pose = 0 #track current pose number
avg_pose_percent_array = [0,0,0,0,0] #track percentages of poses

#callibrate each pose. [pose number, average, remove lowest 20% then get average then * .2]
pose_callibrations = [[0, 12.63, 6.11], [1, 14.53, 6.44], [2, 63.1, 16.15], [3, 41.65, 14.75], [4, 37.21, 12.87], [5, 59.04, 16.88], 
                        [6, 7.07, 4.02], [7, 54.59, 16.19], [8, 48.21, 14.51], [9, 29.25, 9.6], [10, 44.97, 14.56], [11, 28.53, 13.21], 
                        [12, 35.89, 10.71], [13, 23.11, 9.39], [14, 37.74, 12.01], [15, 26.93, 9.38], [16, 24.76, 9.59], [17, 50.48, 14.41], 
                        [18, 38.0, 12.19], [19, 17.19, 6.94], [20, 23.64, 8.16], [21, 32.98, 10.69], [22, 51.73, 15.75], [23, 26.41, 8.27], 
                        [24, 12.07, 7.26], [25, 18.62, 6.94], [26, 56.31, 14.99], [27, 26.09, 11.28], [28, 36.57, 13.04], [29, 28.97, 12.2], 
                        [30, 36.74, 11.78], [31, 48.68, 15.13], [32, 31.01, 10.13], [33, 34.47, 12.09], [34, 7.36, 2.91], [35, 28.1, 11.1], 
                        [36, 34.58, 12.27], [37, 38.97, 13.2], [38, 46.86, 14.45], [39, 38.62, 11.95], [40, 27.0, 9.57], [41, 3.04, 1.53], 
                        [42, 44.36, 14.09], [43, 3.34, 1.64], [44, 50.94, 15.01], [45, 20.29, 10.08], [46, 14.83, 6.11], [47, 14.91, 6.39], 
                        [48, 34.02, 13.87], [49, 15.9, 7.69], [50, 29.25, 10.35], [51, 26.16, 10.94], [52, 24.88, 9.8], [53, 9.95, 4.75], 
                        [54, 48.77, 16.33], [55, 21.29, 7.68], [56, 56.69, 16.57], [57, 51.2, 15.7], [58, 25.36, 10.12], [59, 15.56, 5.54], 
                        [60, 47.75, 14.22], [61, 15.33, 6.43], [62, 10.25, 5.79], [63, 35.89, 11.01], [64, 40.87, 13.11], [65, 33.13, 11.26], 
                        [66, 25.33, 9.26], [67, 4.13, 1.74], [68, 55.61, 15.74], [69, 31.14, 10.4], [70, 9.68, 4.8], [71, 54.95, 16.32], 
                        [72, 17.88, 6.98], [73, 46.57, 13.71], [74, 36.82, 11.81], [75, 57.09, 16.23], [76, 32.81, 10.88], [77, 11.8, 5.56], 
                        [78, 39.38, 13.58], [79, 19.7, 7.48], [80, 38.37, 12.8], [81, 2.16, 0.64]]

#load saved SVC model
filename = 'pose_classifier.pkl'
loaded_model = pickle.load(open(filename,'rb'))

#link to list with txt of pose names
pose_list = "pose_list.txt"
def pose_name(classifier):
    with open(pose_list) as f:
        poses = f.readlines()
    return poses[classifier]

def example_pose_image(pose):
    #example poses image folder
    example_images = 'example_poses'
    target_pose = ''
    for root, dirs, files in os.walk(example_images):
        find_pose = str(pose) + '.jpg'
        for file in files:
            if file == find_pose:
                target_pose = example_images + "/" + find_pose
    return target_pose

#difficulty is from 0-4 for beginner to impossible
def pose_difficulty_selecter(difficulty,poses_to_select):
    #link to list with pose difficulties
    pose_difficulty = "pose_difficulty_list.txt"
    pose_image_links = []
    pose_numbers = []
    with open(pose_difficulty) as f:
        pose_diff_list=[]
        for line in f:
            if line[0] != '#':
                pose_diff_list.append(line.split())
        i = 0
        for poses in pose_diff_list:
            pose_diff_list[i] = sample(poses,poses_to_select)
            i += 1
            
        for pose in pose_diff_list[difficulty]:
            pose_image_links.append(example_pose_image(pose))
            pose_numbers.append(pose)
    print("Pose numbers : ",pose_numbers)
    #Below 3 lines can be used to create a specific routine of poses
    # pose_numbers = ['7','8','9','10','16','22','44','57']
    # pose_image_links = ['example_poses/7.jpg', 'example_poses/8.jpg', 'example_poses/9.jpg', 'example_poses/10.jpg', 'example_poses/16.jpg',
    #                   'example_poses/22.jpg', 'example_poses/44.jpg', 'example_poses/57.jpg']
    # print("Pose image links ", pose_image_links)
    return pose_image_links,pose_numbers

#get probability percent based on target pose and output of SVC
def get_pose_from_landmarks(landmarks,pose):
    print(landmarks_to_save)
    # landmarks_to_save = np.asarray(landmarks_to_save)
    prediction = loaded_model.predict_proba([landmarks_to_save])
    print(prediction)
    pose_probability = prediction[0][pose]
    pose_probability = int(100*pose_probability)
    return pose_probability

#adjust the current pose tracker up or down
def change_pose(up_down):
    global current_pose
    if up_down == 1:
        print("------------changed pose", current_pose)
        if current_pose < (len(current_pose_list) - 1):
            current_pose += 1
        else:
            current_pose = 0
    if up_down == -1:
        print("--------------changed pose",current_pose)
        if current_pose > 0:
            current_pose -= 1
        else:
            current_pose = (len(current_pose_list) - 1)

#calculate pose scores
current_pose_score = []
def calculate_score(current_pose_percentage):
    global current_pose_score
    if timer_started == True:
        current_pose_score.append(current_pose_percentage)
        length = len(current_pose_score)
        total = sum(current_pose_score)
        score = total / length
        grade = 'C'
        if score < 10:
            grade = 'D'
        if score >= 10 and score < 30:
            grade = 'C'
        if score >= 30 and score < 50:
            grade = 'B'
        if score >= 50 and score < 70:
            grade = 'A'
        if score > 85:
            grade = 'S'
        return grade

timer_started = False
start_pose_time = time.time()
#time for each pose
time_per_pose = 15
def countdown_timer(pose_class,avg_percent):
    global start_pose_time
    global timer_started
    if avg_percent > pose_callibrations[pose_class][2] and timer_started == False:
        start_pose_time = time.time()
        timer_started = True
    time_remaining = time_per_pose - (time.time() - start_pose_time)
    print("time_remaining:", time_remaining)
    if time_remaining < 0 and timer_started == True:
        change_pose(1)
        timer_started = False
        time.sleep(1)
    return time_remaining

current_pose_list,current_pose_numbers = pose_difficulty_selecter(0,7)


#################### end new stuff

















while True:
    # Run blazepose on next frame
    landmarks = np.array([])
    frame, body, landmarks = pose.next_frame()
    if frame is None: break
    # Draw 2d skeleton
    frame = renderer.draw(frame, body)
############################# start new stuff ########################
    print("landmarks: ", landmarks)
    print("landmark type", type(landmarks))
    
    #paste example image on input image
    print("example pose : ",current_pose_list[current_pose])
    img = cv2.imread(current_pose_list[current_pose])   
    #adjust example size
    ratio = img.shape[0]/img.shape[1]
    print("ratio = ",ratio)
    example_width = 200
    example_height = int((200 * ratio))
    img = cv2.resize(img,(example_width,example_height),interpolation = cv2.INTER_AREA)
    x_offset = frame.shape[1] - example_width
    y_offset = frame.shape[0]-example_height
    x_end = frame.shape[1]
    y_end = frame.shape[0]
    #put black rectange on right side
    cv2.rectangle(frame,((frame.shape[1]-180),0), (frame.shape[1],frame.shape[0]),(0,0,0),-1)
    frame[y_offset:y_end,x_offset:x_end] = img

    #extract desired landmarks 
    body_part = 0
    desired_pose = int(current_pose_numbers[current_pose])
    print("current pose number : ", desired_pose)
    NoneType = type(None)
    if type(landmarks) != NoneType:
        landmarks_to_save = []
        for poses in landmarks:
            if (body_part > 10 and body_part < 17) or (body_part > 22):
                landmarks_to_save.append(landmarks[body_part][0])
                landmarks_to_save.append(landmarks[body_part][1])
                landmarks_to_save.append(landmarks[body_part][2])
            body_part += 1













    cv2.imshow("demo",frame)

############################# end new stuff ########################




    key = renderer.waitKey(delay=1)
    if key == 27 or key == ord('q'):
        break
renderer.exit()
pose.exit()
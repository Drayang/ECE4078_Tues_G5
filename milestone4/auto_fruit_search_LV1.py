# M4 - Autonomous fruit searching

# basic python packages
import sys, os
import cv2
import numpy as np
import json
import ast
import argparse
import time

# import utility functions
sys.path.insert(0, "util")
from pibot import Alphabot
import measure as measure

# -------------------------------- Newly added imports --------------------------------
from operate import Operate
# -------------------------------- Newly added imports --------------------------------


def read_true_map(fname):
    """Read the ground truth map and output the pose of the ArUco markers and 3 types of target fruit to search

    @param fname: filename of the map
    @return:
        1) list of target fruits, e.g. ['redapple', 'greenapple', 'orange']
        2) locations of the target fruits, [[x1, y1], ..... [xn, yn]]
        3) locations of ArUco markers in order, i.e. pos[9, :] = position of the aruco10_0 marker
    """
    with open(fname, 'r') as f:
        try:
            gt_dict = json.load(f)                   
        except ValueError as e:
            with open(fname, 'r') as f:
                gt_dict = ast.literal_eval(f.readline())   
        fruit_list = []
        fruit_true_pos = []
        aruco_true_pos = np.empty([10, 2])

        # remove unique id of targets of the same type
        for key in gt_dict:
            x = np.round(gt_dict[key]['x'], 1)
            y = np.round(gt_dict[key]['y'], 1)

            if key.startswith('aruco'):
                if key.startswith('aruco10'):
                    aruco_true_pos[9][0] = x
                    aruco_true_pos[9][1] = y
                else:
                    marker_id = int(key[5])
                    aruco_true_pos[marker_id][0] = x
                    aruco_true_pos[marker_id][1] = y
            else:
                fruit_list.append(key[:-2])
                if len(fruit_true_pos) == 0:
                    fruit_true_pos = np.array([[x, y]])
                else:
                    fruit_true_pos = np.append(fruit_true_pos, [[x, y]], axis=0)

        return fruit_list, fruit_true_pos, aruco_true_pos


def read_search_list():
    """Read the search order of the target fruits

    @return: search order of the target fruits
    """
    search_list = []
    with open('search_list.txt', 'r') as fd:
        fruits = fd.readlines()

        for fruit in fruits:
            search_list.append(fruit.strip())

    return search_list


def print_target_fruits_pos(search_list, fruit_list, fruit_true_pos):
    """Print out the target fruits' pos in the search order

    @param search_list: search order of the fruits
    @param fruit_list: list of target fruits
    @param fruit_true_pos: positions of the target fruits
    """

    print("Search order:")
    n_fruit = 1
    for fruit in search_list:
        for i in range(3):
            if fruit == fruit_list[i]:
                print('{}) {} at [{}, {}]'.format(n_fruit,
                                                  fruit,
                                                  np.round(fruit_true_pos[i][0], 1),
                                                  np.round(fruit_true_pos[i][1], 1)))
        n_fruit += 1


# Waypoint navigation
# the robot automatically drives to a given [x,y] coordinate
# additional improvements:
# you may use different motion model parameters for robot driving on its own or driving while pushing a fruit
# try changing to a fully automatic delivery approach: develop a path-finding algorithm that produces the waypoints
def drive_to_point(waypoint, robot_pose):
    # imports camera / wheel calibration parameters 
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')
    
    ####################################################
    # TODO: replace with your codes to make the robot drive to the waypoint
    # One simple strategy is to first turn on the spot facing the waypoint,
    # then drive straight to the way point

    wheel_vel = 10*2.5 # tick to move the robot,*2.5 is to map with the value we commonly put in operate.py slef.command['motion']

    
    # turn towards the waypoint
    turn_time = 0.0 # replace with your calculation
    theta_goal = np.arctan2(waypoint[1]-robot_pose[1], waypoint[0]-robot_pose[0])
    delta_theta = theta_goal - robot_pose[2]
    turn_time = float((abs(delta_theta)*baseline) / (2*wheel_vel*scale))
    print("Turning for {:.2f} seconds".format(turn_time))

    operate.take_pic()
    if delta_theta >0:
        lv,rv = ppi.set_velocity([0, 1], turning_tick=wheel_vel, time=turn_time)
    else:
        lv,rv = ppi.set_velocity([0, -1], turning_tick=wheel_vel, time=turn_time)

    drive_meas = measure.Drive(lv, rv, turn_time)
    operate.update_slam(drive_meas)
    robot_pose = operate.ekf.get_state_vector()[0:3,:]


    
    # after turning, drive straight to the waypoint
    drive_time = 0.0 # replace with your calculation
    delta_dist = ((waypoint[0]-robot_pose[0])**2 + (waypoint[1]-robot_pose[1])**2)**0.5
    drive_time = float(delta_dist / (wheel_vel*scale))
    print("Driving for {:.2f} seconds".format(drive_time))
    operate.take_pic()
    lv, rv = ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)
    
    # Update the slam while moving
    drive_meas = measure.Drive(lv, rv, drive_time)
    operate.update_slam(drive_meas)
    ####################################################

    print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))




def get_robot_pose():
    ####################################################
    # TODO: replace with your codes to estimate the pose of the robot
    # We STRONGLY RECOMMEND you to use your SLAM code from M2 here

    # get robot state
    robot_state = operate.ekf.get_state_vector()

    # update the robot pose [x,y,theta]
    robot_pose = [0.0,0.0,0.0] # replace with your calculation
    robot_pose = robot_state[0:3,:]
    ####################################################

    return robot_pose

######################## REPLACE WITH OUR OWN CODE #########################
# To rotate the robot slowly until it scan a aruco marker at the camera centre frame
# and update robot current pose based on the aruco marker location
# call it whenever reacha way point
def recentre():
    
    return None

# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M4_true_map.txt')
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=8000)
    
    ##################### REPLACE WITH OWN CODE #####################
    
    # For creating Operate class purpose
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--ckpt", default='network/scripts/model/model.best.pth')

    ##################### REPLACE WITH OWN CODE #####################
    

    args, _ = parser.parse_known_args()
    ppi = Alphabot(args.ip,args.port)


    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

    waypoint = [0.0,0.0]
    robot_pose = [0.0,0.0,0.0]

    ############## REPLACE WITH OWN CODE #####################
    operate = Operate(args)
    ############## REPLACE WITH OWN CODE #####################

    # The following code is only a skeleton code the semi-auto fruit searching task
    while True:
        # enter the waypoints
        # instead of manually enter waypoints in command line, you can get coordinates by clicking on a map (GUI input), see camera_calibration.py
        x,y = 0.0,0.0
        x = input("X coordinate of the waypoint: ")
        try:
            x = float(x)
        except ValueError:
            print("Please enter a number.")
            continue
        y = input("Y coordinate of the waypoint: ")
        try:
            y = float(y)
        except ValueError:
            print("Please enter a number.")
            continue


        ############## REPLACE WITH OWN CODE #####################
        # run SLAM (copy from operate.py update_keyboard() function)
        n_observed_markers = len(operate.ekf.taglist)
        if n_observed_markers == 0:
            if not operate.ekf_on:
                print('SLAM is running')
                operate.ekf_on = True
            else:
                print('> 2 landmarks is required for pausing')
        elif n_observed_markers < 3:
            print('> 2 landmarks is required for pausing')
        else:
            if not operate.ekf_on:
                operate.request_recover_robot = True
            operate.ekf_on = not operate.ekf_on
            if operate.ekf_on:
                print('SLAM is running')
            else:
                print('SLAM is paused')

        ############## REPLACE WITH OWN CODE #####################

        
        # estimate the robot's pose
        robot_pose = get_robot_pose()

        # robot drives to the waypoint
        waypoint = [x,y]
        drive_to_point(waypoint,robot_pose) ###### add return to drive_to_point function to get updatee pose
        robot_pose = get_robot_pose()
        print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))


        # exit
        ppi.set_velocity([0, 0])
        uInput = input("Add a new waypoint? [Y/N]")
        if uInput == 'N':
            break
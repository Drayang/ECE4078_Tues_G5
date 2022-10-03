# M4 - Autonomous fruit searching

# basic python packages
import sys, os
import cv2
import numpy as np
import json
import ast
import argparse
import time

# import SLAM components
# sys.path.insert(0, "{}/slam".format(os.getcwd()))
# from slam.ekf import EKF
# from slam.robot import Robot
# import slam.aruco_detector as aruco


# import utility functions
sys.path.insert(0, "util")
from pibot import Alphabot
import measure as measure

##################### REPLACE WITH OWN CODE #####################
from operate import Operate
import pygame # for GUI

#---------------------------- For path planning ------------------------------#
sys.path.insert(0, "{}/path_planning".format(os.getcwd()))
import matplotlib.pyplot as plt
import math
from path_planning.dijkstra import Dijkstra
from path_planning.a_star import AStarPlanner

from path_planning.lqr_rrt_star import LQRRRTStar
#---------------------------- For path planning ------------------------------#

##################### REPLACE WITH OWN CODE #####################


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
                    aruco_true_pos[marker_id-1][0] = x
                    aruco_true_pos[marker_id-1][1] = y
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

    
    # # turn towards the waypoint
    # turn_time = 0.0 # replace with your calculation
    # theta_goal = np.arctan2(waypoint[1]-robot_pose[1], waypoint[0]-robot_pose[0])
    # delta_theta = theta_goal - robot_pose[2]

    # # to handle robot turn 360 degree issue
    # # if delta_theta>np.pi:
    # #     delta_theta-=np.pi*2
    # # elif delta_theta<-np.pi:
    # #    delta_theta+=np.pi*2


    # turn_time = float((abs(delta_theta)*baseline) / (2*wheel_vel*scale))
    # print("Turning for {:.2f} seconds".format(turn_time))

    
    # if delta_theta == 0: # To handle 0 turning case
    #     print("Not turning required!")

    # else:
    #     # operate.take_pic()
    #     if delta_theta >0:
    #         self_update_slam([0,1],wheel_vel,turn_time)
    #         # lv,rv = ppi.set_velocity([0, 1], turning_tick=wheel_vel, time=turn_time)
    #     else:
    #         self_update_slam([0,-1],wheel_vel,turn_time)
    #         # lv,rv = ppi.set_velocity([0, -1], turning_tick=wheel_vel, time=turn_time)
        
    #     # drive_meas = measure.Drive(lv, rv, turn_time)
    #     # operate.update_slam(drive_meas)
    #     # robot_pose = operate.ekf.get_state_vector()[0:3,:]

    # robot_pose = get_robot_pose()
    
    # after turning, drive straight to the waypoint
    drive_time = 0.0 # replace with your calculation
    delta_dist = ((waypoint[0]-robot_pose[0])**2 + (waypoint[1]-robot_pose[1])**2)**0.5
    drive_time = float(delta_dist / (wheel_vel*scale))
    print("Driving for {:.2f} seconds".format(drive_time))
    self_update_slam([1,0],wheel_vel,drive_time)
    # operate.take_pic()
    # lv, rv = ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)
    
    # # Update the slam while moving
    # drive_meas = measure.Drive(lv, rv, drive_time)
    # operate.update_slam(drive_meas)
    ####################################################

    print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))


def turn_to_point(waypoint,robot_pose):
    #  imports camera / wheel calibration parameters 
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')
    
    ####################################################
    # TODO: replace with your codes to make the robot drive to the waypoint
    # One simple strategy is to first turn on the spot facing the waypoint,
    # then drive straight to the way point

    wheel_vel = 10*2.0 # tick to move the robot,*2.5 is to map with the value we commonly put in operate.py slef.command['motion']

    
    # turn towards the waypoint
    turn_time = 0.0 # replace with your calculation
    theta_goal = np.arctan2(waypoint[1]-robot_pose[1], waypoint[0]-robot_pose[0])
    delta_theta = theta_goal - robot_pose[2]

    # to handle robot turn 360 degree issue
    if delta_theta>np.pi:
        delta_theta-=np.pi*2
    elif delta_theta<-np.pi:
       delta_theta+=np.pi*2

    turn_time = float((abs(delta_theta)*baseline) / (2*wheel_vel*scale))
    print("Turning for {:.2f} seconds".format(turn_time))

    
    if delta_theta == 0: # To handle 0 turning case
        print("Not turning required!")

    else:
        if delta_theta >0:
            self_update_slam([0,1],wheel_vel,turn_time)
        else:
            self_update_slam([0,-1],wheel_vel,turn_time)

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


#--------------------------------------- For path planning-------------------------------------#
def initialise_space(fruits_true_pos,aruco_true_pos,search_order):
    ox,oy=[],[] # obstacle location

    # to get the fruit idx based on the search list
    for i in range(3):
        if search_list[search_order] == fruits_list[i]:
            search_idx = i
    # define the obstacle location
    for i in range(3):
        if i == search_idx: # do not include the current fruit goal as obstacle
            continue
        ox.append(fruits_true_pos[i][0])
        oy.append(fruits_true_pos[i][1])
    for i in range(10):
        ox.append(aruco_true_pos[i][0])
        oy.append(aruco_true_pos[i][1])

    print("Number of obstacle is : ",len(ox))

    return ox,oy

# search order tell us which fruit we going to move to now
def path_planning(search_order):

    fileB = "calibration/param/baseline.txt"
    robot_radius = np.loadtxt(fileB, delimiter=',')*2 # robot radius = baseline of the robot/2.0
    robot_radius = 0.2
    robot_pose = get_robot_pose() # estimate the robot's pose
    print("Search order is:", search_order)
    sx,sy = float(robot_pose[0]),float(robot_pose[1]) # starting location
    # gx,gy = fruits_true_pos[search_order][0],fruits_true_pos[search_order][1] # goal position

    for i in range(3): # to get the correct fruit idx based on the search list
        if search_list[search_order] == fruits_list[i]:
            gx,gy = fruits_true_pos[i][0],fruits_true_pos[i][1] # goal position

    print("starting loation is: ",sx,",",sy)
    print("ending loation is: ",gx,",",gy)
    
#--------------------------------------- Using Dijkstra-------------------------------------#
    if True:  # pragma: no cover
        plt.figure(figsize=(9,9))
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("square")

    grid_size = 0.2
    dijkstra = Dijkstra(ox, oy, grid_size, robot_radius)
    rx, ry = dijkstra.planning(sx, sy, gx, gy)
    
    for i in range(len(rx)):
        rx[i]= round(rx[i],2)
        ry[i]= round(ry[i],2)


    print("The x path is:",rx)
    print("The y path is:",ry)
    print("The last location is:",rx[0],",",ry[0])
    

    if True:  # pragma: no cover
        ticks=np.arange(-1.6,1.6,0.4)
        plt.xticks(ticks)
        plt.yticks(ticks)
        plt.plot(rx[1:], ry[1:], "-r")
        plt.xlim([-1.6,1.6])
        plt.ylim([-1.6,1.6])
        plt.tick_params(length=0, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
        plt.savefig(f'./pics/map_grid.jpg', bbox_inches='tight',transparent=True, pad_inches=0)
        

#--------------------------------------- Using Dijkstra-------------------------------------#
#--------------------------------------- Using AStar-------------------------------------#
    # if True:  # pragma: no cover
    #     plt.plot(ox, oy, ".k")
    #     plt.plot(sx, sy, "og")
    #     plt.plot(gx, gy, "xb")
    #     plt.grid(True)
    #     plt.axis("equal")

    # grid_size = 0.20

    # a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
    # rx, ry = a_star.planning(sx, sy, gx, gy)


    # if True:  # pragma: no cover
    #     plt.plot(rx, ry, "-r")
    #     plt.pause(0.001)
    #     plt.show()
#--------------------------------------- Using AStar-------------------------------------#

    return rx,ry


######################## REPLACE WITH OUR OWN CODE #########################
#update the slam + take picture
def self_update_slam(command,wheel_vel,turn_time):
    
    if not (command[0] == 0 and command[1] == 0): # skip stop ([0,0]) command
        if command[0] == 0: # turning command
            lv,rv = ppi.set_velocity(command, turning_tick=wheel_vel, time=turn_time)    
        else: # moving straight command
            lv,rv = ppi.set_velocity(command, tick=wheel_vel, time=turn_time)    

        drive_meas = measure.Drive(lv, rv, turn_time)
        operate.take_pic()
        operate.update_slam(drive_meas)
        self_update_GUI()

def self_update_GUI():
    operate.draw(canvas)
    pygame.display.update()
######################## REPLACE WITH OUR OWN CODE #########################

# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M4_true_map.txt')
    parser.add_argument("--ip", metavar='', type=str, default='192.168.137.206')
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
    print("SearchList is:",search_list)
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

    waypoint = [0.0,0.0]
    robot_pose = [0.0,0.0,0.0]


    search_order = 0 # indicate search which fruit now

    ############## REPLACE WITH OWN CODE #####################
    operate = Operate(args)

        # The following code is only a skeleton code the semi-auto fruit searching task
    pygame.font.init()     
    
    width, height = 902, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 2022 Lab')
    pygame.display.set_icon(pygame.image.load('pics/8bit/pibot5.png'))
    canvas.fill((0, 0, 0))
    splash = pygame.image.load('pics/loading.png')
    pibot_animate = [pygame.image.load('pics/8bit/pibot1.png'),
                     pygame.image.load('pics/8bit/pibot2.png'),
                     pygame.image.load('pics/8bit/pibot3.png'),
                    pygame.image.load('pics/8bit/pibot4.png'),
                     pygame.image.load('pics/8bit/pibot5.png')]
    pygame.display.update()

    start = False

    counter = 40
    while not start:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                start = True
        canvas.blit(splash, (0, 0))
        x_ = min(counter, 600)
        if x_ < 600:
            canvas.blit(pibot_animate[counter%10//2], (x_, 565))
            pygame.display.update()
            counter += 2



  
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

    # update SLAM (TODO: can replace with self_update_slam() function)
    operate.take_pic()
    lv,rv = ppi.set_velocity([0, 0], tick=0.0, time=0.0)
    drive_meas = measure.Drive(lv, rv, 0.0)
    operate.update_slam(drive_meas)

    lms=[]
    for i,lm in enumerate(aruco_true_pos):
        measure_lm = measure.Marker(np.array([[lm[0]],[lm[1]]]),i+1)
        lms.append(measure_lm)
    operate.ekf.add_landmarks(lms)   

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


        # estimate the robot's pose
        robot_pose = get_robot_pose()

        # robot drives to the waypoint
        waypoint = [x,y]
        drive_to_point(waypoint,robot_pose) ###### add return to drive_to_point function to get updatee pose
        robot_pose = get_robot_pose()
        print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))


        ############## REPLACE WITH OWN CODE #####################
        # update SLAM again
        operate.take_pic()
        lv,rv = ppi.set_velocity([0, 0], turning_tick=0.0, time=0.0)
        drive_meas = measure.Drive(lv, rv, 0.0)
        operate.update_slam(drive_meas)

        ############## REPLACE WITH OWN CODE #####################


        # exit
        ppi.set_velocity([0, 0])
        uInput = input("Add a new waypoint? [Y/N]")
        if uInput == 'N' or uInput == 'n':
            break
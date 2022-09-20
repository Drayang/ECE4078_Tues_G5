import csv
import math


def radians(short):
    if short == 1:
        return math.pi/2
    elif short == 2:
        return math.pi/4
    elif short == 3:
        return 0
    elif short == 4:
        return -1 * math.pi/4
    elif short == 5:
        return -1 * math.pi/2
    elif short == 6:
        return -1 * 3/4 * math.pi
    elif short == 7:
        return math.pi
    elif short == 8:
        return 3/4 * math.pi

rows = []

with open('robot_pose.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        rows.append(row)

i = 0
with open("images.txt", 'w') as f:
    for row in rows[1:]:
        angle = radians(int(row[2]))
        f.write(f'{{"pose": [[{round(float(row[0])*0.4,1)}], [{round(float(row[1])*0.4,1)}], [{angle}]], "imgfname": "lab_output/pred_{i}.png"}}\n')
        i = i + 1
    
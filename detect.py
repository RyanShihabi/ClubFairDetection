import cv2
import csv
import numpy as np
import matplotlib.patches as patches
import pandas as pd
import torch

person_model = torch.hub.load("ultralytics/yolov5", "custom", path="crowdhuman_yolov5m.pt")

headers = ["second", "Q1 Count", "Q2 Count", "Q3 Count", "Q4 Count"]
rows = []
quadrant_counts = {1: 0, 2: 0, 3: 0, 4: 0}

# Initial start time: 10:38 AM
# End time: 1:28:41 PM
# cap = cv2.VideoCapture("Engagement_Fair_2022.MOV")
cap = cv2.VideoCapture("cut.mp4")

canvas = np.zeros((1080, 1920, 3), np.uint8)
g_ellipse = patches.Ellipse((1060, 470), 425, 200, angle=360, fill=False)

seconds = 0
sub_sections = 4
frame_count = 0

# TODO
# Segmentation for fountain and entire piazza
# head count for area near Beckman: rest is good to go

while cap.isOpened():
    ret, frame = cap.read()
 
    frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    if ret:
        if frame_count % 60 == 0:
            second_total_q1 = 0
            second_total_q2 = 0
            second_total_q3 = 0
            second_total_q4 = 0

            canvas = np.zeros((1080, 1920, 3), np.uint8)
            
            frame = cv2.ellipse(frame, (1060, 470), (425, 200), 0, 0, 360, (0, 0, 255), 3)
            frame = cv2.ellipse(frame, (960, 600), (960, 455), 0, 0, 360, (255, 0, 0), 3)
            canvas = cv2.ellipse(canvas, (1060, 470), (425, 200), 0, 0, 360, (0, 0, 255), 3)
            canvas = cv2.ellipse(canvas, (960, 600), (960, 455), 0, 0, 360, (255, 0, 0), 3)

            # quadrant split
            for i in range(1, sub_sections+1):
                for j in range(1, sub_sections+1):  
                    min_bound_x = int(frame.shape[1]*((j-1)/sub_sections))
                    max_bound_x = int(frame.shape[1]*(j/sub_sections))

                    min_bound_y = int(frame.shape[0]*((i-1)/sub_sections))
                    max_bound_y = int(frame.shape[0]*(i/sub_sections))

                    quadrant = frame[min_bound_y:max_bound_y, min_bound_x:max_bound_x]

            # x axis split
            # for i in range(1, sub_sections+1):
            #     min_bound_x = int(frame.shape[1]*(max(0,i-1)/sub_sections))
            #     max_bound_x = int(frame.shape[1]*(i/sub_sections))

                # quadrant = frame[:, min_bound_x:max_bound_x]

                # print(min_bound_x, ":", max_bound_x)

                    inference = person_model(quadrant)

                    if not inference.pandas().xyxy[0].empty:
                        for index, row in inference.pandas().xyxy[0].iterrows():
                            if(row["name"] == "head"):
                                cv2.putText(frame, str(row["confidence"])[:5], (int(row["xmax"]+min_bound_x), int(row["ymin"])-10+min_bound_y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                                cv2.rectangle(frame, (int(row["xmin"]+min_bound_x), int(row["ymin"])+min_bound_y), (int(row["xmax"]+min_bound_x), int(row["ymax"])+min_bound_y), (0, 255, 0), 2)

                                center_x = int((row["xmin"] + row["xmax"])/2) + min_bound_x
                                # center_y = int((row["ymin"] + row["ymax"])/2)

                                center_y = int((row["ymin"] + row["ymax"])/2) + min_bound_y

                                if center_x >= int(frame.shape[1]/2) and center_y < int(frame.shape[0]/2):
                                    second_total_q1 += 1
                                elif center_x < int(frame.shape[1]/2) and center_y < int(frame.shape[0]/2):
                                    second_total_q2 += 1
                                elif center_x < int(frame.shape[1]/2) and center_y >= int(frame.shape[0]/2):
                                    second_total_q3 += 1
                                else:
                                    second_total_q4 += 1

                                cv2.circle(canvas, (center_x, center_y), 10, (0, 255, 0), -1)

            rows.append({
                "second": seconds,
                "Q1 Count": second_total_q1,
                "Q2 Count": second_total_q2,
                "Q3 Count": second_total_q3,
                "Q4 Count": second_total_q4
            })

            seconds += 1

            cv2.imshow("frame", frame)
            cv2.imshow("canvas", canvas)
    else:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

with open('club_fair.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(rows)
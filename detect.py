import cv2
import numpy as np
import matplotlib.patches as patches
import pandas as pd
import torch

person_model = torch.hub.load("ultralytics/yolov5", "custom", path="crowdhuman_yolov5m.pt")
# person_model = torch.load("./e2edet_final.pth", map_location=torch.device('cpu'))

cap = cv2.VideoCapture("Engagement_Fair_2022.MOV")

canvas = np.zeros((1080, 1920, 3), np.uint8)

g_ellipse = patches.Ellipse((1060, 470), 425, 200, angle=360, fill=False)

while cap.isOpened():
    ret, frame = cap.read()
 
    # Select ROI
    # r = cv2.selectROI(frame)
 
    # # Crop image
    # imCrop = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    # print((int(r[1]), int(r[1]+r[3])), (int(r[0]), int(r[0]+r[2])))
 
    # # Display cropped image
    # cv2.imshow("Image", imCrop)
    # cv2.waitKey(0)

    # quadrant 1
    # frame_q1 = frame[130:470, 1050:1900]

    # quadrant 4
    # frame_q4 = frame[130:470, 26:1050]

    # frame = cv2.resize(frame_bgr, (frame_bgr.shape[1]*3, frame_bgr.shape[0]*3))
    frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    if ret:
        canvas = np.zeros((1080, 1920, 3), np.uint8)
        # TODO
        # Segmentation for fountain and entire piazza
        frame = cv2.ellipse(frame, (1060, 470), (425, 200), 0, 0, 360, (0, 0, 255), 3)
        frame = cv2.ellipse(frame, (960, 600), (960, 455), 0, 0, 360, (255, 0, 0), 3)
        canvas = cv2.ellipse(canvas, (1060, 470), (425, 200), 0, 0, 360, (0, 0, 255), 3)
        canvas = cv2.ellipse(canvas, (960, 600), (960, 455), 0, 0, 360, (255, 0, 0), 3)

        # frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # quadrant 1
        frame_q1 = frame[130:470, 1050:1900]
        # frame_q1_q1 = frame_q1[:frame_q1.shape[0]/2, frame_q1.shape[1]/2:]
        # frame_q1_q2 = frame_q1[frame_q1.shape[0]/2:, frame_q1.shape[1]/2:]
        # frame_q1_q3 = frame_q1[frame_q1.shape[0]/2:, :frame_q1.shape[1]/2]
        # frame_q1_q4 = frame_q1[:frame_q1.shape[0]/2, :frame_q1.shape[1]/2]

        # # quadrant 2
        frame_q2 = frame[470:1080, 1050:1900]
        # frame_q2_q1 = frame_q2[:frame_q2.shape[0]/2, frame_q2.shape[1]/2:]
        # frame_q2_q2 = frame_q2[frame_q2.shape[0]/2:, frame_q2.shape[1]/2:]
        # frame_q2_q3 = frame_q2[frame_q2.shape[0]/2:, :frame_q2.shape[1]/2]
        # frame_q2_q4 = frame_q2[:frame_q2.shape[0]/2, :frame_q2.shape[1]/2]

        # # quadrant 3
        frame_q3 = frame[470:1080, 26:1050]
        # frame_q3_q1 = frame_q3[:frame_q3.shape[0]/2, frame_q3.shape[1]/2:]
        # frame_q3_q2 = frame_q3[frame_q3.shape[0]/2:, frame_q3.shape[1]/2:]
        # frame_q3_q3 = frame_q3[frame_q3.shape[0]/2:, :frame_q3.shape[1]/2]
        # frame_q3_q4 = frame_q3[:frame_q3.shape[0]/2, :frame_q3.shape[1]/2]

        # quadrant 4
        frame_q4 = frame[130:470, 26:1050]
        # frame_q4_test = frame[130:470, 100:525]
        # frame_q4_q1 = frame_q4[:int(frame_q4.shape[0]/2), int(frame_q4.shape[1]/2):]
        # frame_q4_q2 = frame_q4[int(frame_q4.shape[0]/2):, int(frame_q4.shape[1]/2):]
        # frame_q4_q3 = frame_q4[int(frame_q4.shape[0]/2):, :int(frame_q4.shape[1]/2)]
        # frame_q4_q4 = frame_q4[:int(frame_q4.shape[0]/2), :int(frame_q4.shape[1]/2)]
        # frame_bgr = cv2.resize(frame[:400, :400])

        # frame_bgr = cv2.resize(frame_bgr, (frame_bgr.shape[1]*3, frame_bgr.shape[0]*3))

        # results = person_model(frame_bgr)

        results_q1 = person_model(frame_q1)
        results_q2 = person_model(frame_q2)
        results_q3 = person_model(frame_q3)
        results_q4 = person_model(frame_q4)

        # results_q4_q1 = person_model(frame_q4_q1)
        # results_q4_q2 = person_model(frame_q4_q2)
        # results_q4_q3 = person_model(frame_q4_q3)
        # results_q4_q4 = person_model(frame_q4_q4)

        # if not results_q4_q1.pandas().xyxy[0].empty:
        #     for index, row in results_q4_q1.pandas().xyxy[0].iterrows():
        #         if(row["name"] == "person" and row["confidence"] > 0.4):
        #             cv2.putText(frame, str(row["confidence"])[:5], (int(row["xmax"]+26), int(row["ymin"])-10+130), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        #             cv2.rectangle(frame, (int(row["xmin"]+130), int(row["ymin"]+130)), (int(row["xmax"]+26), int(row["ymax"]+130)), (0, 255, 0), 2)
        #             center_x = int((row["xmin"] + row["xmax"])/2)
        #             center_y = int((row["ymin"] + row["ymax"])/2)
        #             # numpy check circle areas
        #             cv2.circle(canvas, (center_x, center_y), 10, (0, 255, 0), -1)
        
        # if not results_q4_q2.pandas().xyxy[0].empty:
        #     for index, row in results_q4_q2.pandas().xyxy[0].iterrows():
        #         if(row["name"] == "person" and row["confidence"] > 0.4):
        #             cv2.putText(frame, str(row["confidence"])[:5], (int(row["xmax"]+26), int(row["ymin"])-10+130), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        #             cv2.rectangle(frame, (int(row["xmin"]+26), int(row["ymin"]+130)), (int(row["xmax"]+26), int(row["ymax"]+130)), (0, 255, 0), 2)
        #             center_x = int((row["xmin"] + row["xmax"])/2) + 26
        #             center_y = int((row["ymin"] + row["ymax"])/2) + 130
        #             # numpy check circle areas
        #             cv2.circle(canvas, (center_x, center_y), 10, (0, 255, 0), -1)
        
        # if not results_q4_q3.pandas().xyxy[0].empty:
        #     for index, row in results_q4_q3.pandas().xyxy[0].iterrows():
        #         if(row["name"] == "person" and row["confidence"] > 0.4):
        #             cv2.putText(frame, str(row["confidence"])[:5], (int(row["xmax"]+26), int(row["ymin"])-10+130), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        #             cv2.rectangle(frame, (int(row["xmin"]+26), int(row["ymin"]+130)), (int(row["xmax"]+26), int(row["ymax"]+130)), (0, 255, 0), 2)
        #             center_x = int((row["xmin"] + row["xmax"])/2) + 26
        #             center_y = int((row["ymin"] + row["ymax"])/2) + 130
        #             # numpy check circle areas
        #             cv2.circle(canvas, (center_x, center_y), 10, (0, 255, 0), -1)

        # if not results_q4_q4.pandas().xyxy[0].empty:
        #     for index, row in results_q4_q4.pandas().xyxy[0].iterrows():
        #         if(row["name"] == "person" and row["confidence"] > 0.4):
        #             cv2.putText(frame, str(row["confidence"])[:5], (int(row["xmax"]+26), int(row["ymin"])-10+130), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        #             cv2.rectangle(frame, (int(row["xmin"]+26), int(row["ymin"]+130)), (int(row["xmax"]+26), int(row["ymax"]+130)), (0, 255, 0), 2)
        #             center_x = int((row["xmin"] + row["xmax"])/2) + 26
        #             center_y = int((row["ymin"] + row["ymax"])/2) + 130
        #             # numpy check circle areas
        #             cv2.circle(canvas, (center_x, center_y), 10, (0, 255, 0), -1)
        # print(results)

        # q1_results = person_model(frame_bgr[470:800, 200:425])

        if not results_q1.pandas().xyxy[0].empty:
            for index, row in results_q1.pandas().xyxy[0].iterrows():
                if(row["name"] == "person" and row["confidence"] > 0.4):
                    cv2.putText(frame, str(row["confidence"])[:5], (int(row["xmax"]+1050), int(row["ymin"])-10+130), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.rectangle(frame, (int(row["xmin"]+1050), int(row["ymin"]+130)), (int(row["xmax"]+1050), int(row["ymax"]+130)), (0, 255, 0), 2)
                    center_x = int((row["xmin"] + row["xmax"])/2) + 1050
                    center_y = int((row["ymin"] + row["ymax"])/2) + 130
                    # numpy check circle areas
                    cv2.circle(canvas, (center_x, center_y), 10, (0, 255, 0), -1)

        if not results_q2.pandas().xyxy[0].empty:
            for index, row in results_q2.pandas().xyxy[0].iterrows():
                if(row["name"] == "person" and row["confidence"] > 0.4):
                    cv2.putText(frame, str(row["confidence"])[:5], (int(row["xmax"]+1050), int(row["ymin"])-10+470), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.rectangle(frame, (int(row["xmin"]+1050), int(row["ymin"]+470)), (int(row["xmax"]+1050), int(row["ymax"]+470)), (0, 255, 0), 2)
                    center_x = int((row["xmin"] + row["xmax"])/2) + 1050
                    center_y = int((row["ymin"] + row["ymax"])/2) + 470
                    # numpy check circle areas
                    cv2.circle(canvas, (center_x, center_y), 10, (0, 255, 0), -1)

        if not results_q3.pandas().xyxy[0].empty:
            for index, row in results_q3.pandas().xyxy[0].iterrows():
                if(row["name"] == "person" and row["confidence"] > 0.4):
                    cv2.putText(frame, str(row["confidence"])[:5], (int(row["xmax"]+26), int(row["ymin"])-10+470), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.rectangle(frame, (int(row["xmin"]+26), int(row["ymin"]+470)), (int(row["xmax"]+26), int(row["ymax"]+470)), (0, 255, 0), 2)
                    center_x = int((row["xmin"] + row["xmax"])/2) + 26
                    center_y = int((row["ymin"] + row["ymax"])/2) + 470
                    # numpy check circle areas
                    cv2.circle(canvas, (center_x, center_y), 10, (0, 255, 0), -1)

        if not results_q4.pandas().xyxy[0].empty:
            for index, row in results_q4.pandas().xyxy[0].iterrows():
                if(row["name"] == "person" and row["confidence"] > 0.4):
                    cv2.putText(frame, str(row["confidence"])[:5], (int(row["xmax"]+26), int(row["ymin"])-10+130), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.rectangle(frame, (int(row["xmin"]+26), int(row["ymin"]+130)), (int(row["xmax"]+26), int(row["ymax"]+130)), (0, 255, 0), 2)
                    center_x = int((row["xmin"] + row["xmax"])/2) + 26
                    center_y = int((row["ymin"] + row["ymax"])/2) + 130
                    # numpy check circle areas
                    cv2.circle(canvas, (center_x, center_y), 10, (0, 255, 0), -1)

        cv2.imshow("frame", frame)
        cv2.imshow("canvas", canvas)
    else:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
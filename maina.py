from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

import cv2
from ultralytics import YOLO
import torch

import util
import pandas as pd
from sort.sort import *
from util import get_car, read_license_plate, write_csv

torch.cuda.set_device(0)

results = {}

mot_tracker = Sort()

coco_model = YOLO("yolov8n.pt")
license_plate_model = YOLO("./model/best.pt")

# 2 - car, 3 - motorcycle, 5 - bus, 7 - truck
vehicles = [2,3,5,7]

uri = "mongodb+srv://ragi:q1w2e3r4.!@cluster0.tys9hdc.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["licenseToSteal"]
# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

locC = db["location"]
licC = db["licenseplates"]

# Starts camera capture (Live Feed)
# vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Video Upload
vid = cv2.VideoCapture("demo.mp4")

output = cv2.VideoWriter(
        "output.avi", cv2.VideoWriter_fourcc(*'MPEG'),
      30, (1080, 1920))

# Only used if uploading video
frame_nmr = -1

ret = True
while ret:
    frame_nmr += 1
    ret, frame = vid.read()
    if ret:
        results[frame_nmr] = {}

        detections = coco_model(frame, device='gpu')[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
               detections_.append([x1, y1, x2, y2, score])
               cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

        if len(detections_) > 0:
            track_ids = mot_tracker.update(np.asarray(detections_))
        else:
            track_ids = mot_tracker.update(np.empty((0, 5)))

        license_plates = license_plate_model(frame, device='gpu')[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                          (255, 0, 0), 3)
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 128, 255, cv2.THRESH_BINARY_INV)
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                print(license_plate_text)
                if license_plate_text is not None:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, license_plate_text, (int(x1), int(y1)), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_scor'
                                                                    ''
                                                                    'e': score,
                                                                    'text_score': license_plate_text_score}}
                    if score > 0.88:
                        q = {"number": license_plate_text}
                        plate = list(licC.find(q))
                        if len(plate) > 0:
                            if plate[0]['stolen']:
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                cv2.putText(frame, "STOLEN VEHICLE DETECTED", (int(x1-30), int(y1+80)), font, 4, (255, 0, 0), 3,
                                            cv2.LINE_AA)
        output.write(frame)
        cv2.imshow('output', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    else:
        break

write_csv(results, './license_plates.csv')
cv2.destroyAllWindows()
output.release()
vid.release()
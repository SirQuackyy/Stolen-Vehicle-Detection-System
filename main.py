from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from math import radians, degrees, sin, cos, asin, acos, sqrt

import customtkinter
import cv2
import util
from ultralytics import YOLO
from PIL import Image, ImageTk
import cv2 as cv
import torch

from sort.sort import *
from util import get_car, read_license_plate, write_csv

torch.cuda.set_device(0)

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

customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("dark-blue")
cap = cv.VideoCapture(0)
# cap = cv2.VideoCapture("demo.mp4")

results = {}
mot_tracker = Sort()

coco_model = YOLO("yolov8n.pt")
license_plate_model = YOLO("./model/best.pt")


vehicles = [2, 3, 5, 7]


def great_circle(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    return 6371 * (
        acos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon1 - lon2))
    )

class Video(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("1900x700")
        self.grid_rowconfigure(0, weight=1)  # configure grid system
        self.grid_columnconfigure(0, weight=1)

        self.label = customtkinter.CTkLabel(self)

        self.label.grid(row=0, column=0, padx=20)

        self.my_frame = self.label
        self.my_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        self.bind('<Escape>', lambda e: self.quit())
        self.frame_nmr = -1

    def CameraInFrame(self):
        width, height = 800, 700
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.frame_nmr += 1

        ret = True

        if ret:
            ret, frame = cap.read()
            results[self.frame_nmr] = {}
            detections = coco_model(frame, device='gpu')[0]
            detections_ = []
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in vehicles:
                    detections_.append([x1, y1, x2, y2, score])
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                  (0, 255, 0), 3)

            if len(detections_) > 0:
                track_ids = mot_tracker.update(np.asarray(detections_))
            else:
                track_ids = mot_tracker.update(np.empty((0, 5)))

            license_plates = license_plate_model(frame, device='gpu')[0]
            for plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = plate
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(plate, track_ids)

                for license_plate in license_plates.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = license_plate
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                  (255, 0, 0), 3)
                    xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                    #if car_id != -1 or car_id >= 0:
                    license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 128, 255,
                                                                 cv2.THRESH_BINARY_INV)

                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                    if license_plate_text is not None:
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(frame, license_plate_text, (int(x1), int(y1)), font, 2, (255, 255, 255), 2,
                                    cv2.LINE_AA)

                        if score > 0.88:
                            q = {"number": license_plate_text}
                            plate = list(licC.find(q))
                            if len(plate) > 0:
                                if plate[0]['stolen']:
                                    font = cv2.FONT_HERSHEY_SIMPLEX
                                    cv2.putText(frame, "STOLEN VEHICLE DETECTED", (int(x1 - 30), int(y1 + 80)), font, 4,
                                                (255, 0, 0), 3,
                                                cv2.LINE_AA)
                                    lat = 37.4108
                                    long = 122.0311
                                    prev = list(locC.find({"number": license_plate_text}))
                                    isNear = False
                                    for x in prev:
                                        if great_circle(long, lat, x['long'], x['lat']) < 1:
                                            isNear = True
                                            break
                                    if not isNear:
                                        locC.insert_one({"number": license_plate_text, "long": long, "lat": lat, "val": 1})

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            dis_image = Image.fromarray(image)
            pho_image = customtkinter.CTkImage(dis_image, size=(width,height))
            self.label.pho_image = pho_image
            self.label.configure(image=pho_image)
            self.after(20, self.CameraInFrame)


if __name__ == "__main__":
    app = Video()
    app.CameraInFrame()

    app.mainloop()

    cap.release()


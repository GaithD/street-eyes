import json
from dataclasses import asdict
from itertools import combinations

import cv2
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
from scipy.spatial.distance import euclidean
from ultralytics import YOLO
import time
import numpy as np
import csv
import subprocess
import math

from models import ObjectType, CameraData, RecordsByTime, load_data

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8x.pt')  # Use pre-trained YOLOv8 model


# The URL of the image feed (replace with actual URL)
# image_url = "https://webcams.nyctmc.org/api/cameras/5352e130-4668-4be5-a7b9-9e1ce4ea6d4c/image?t=1733625086520"


def euclidean_distance(box1, box2):
    # Calculate the center points
    x1, y1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
    x2, y2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# item['area'] == 'Manhattan' and
def get_all_cameras():
    url = "https://webcams.nyctmc.org/api/cameras/"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        manhatan = [item for item in data if item['area'] == 'Manhattan' and item['isOnline'] == 'true']
        # print(manhatan)
        return manhatan


def fetch_image(url):
    """Fetch the latest image from a URL."""
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            if image:
                return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except OSError as os_err:
        print(f"Image processing error: {os_err}")
    return None


def get_all_objects(frame, known_parking, types=[ObjectType.PERSON.value, ObjectType.CAR.value], cof=0.25,
                    iou=0.45) -> CameraData:
    """Detect and count people, and draw bounding boxes around them."""
    results = model(frame, conf=cof, iou=iou, device='mps')  # Run detection on the frame
    # Loop over all detected boxes and count those that correspond to "person"
    camera_data = CameraData()
    start_time = int(time.time())

    color = (0, 0, 255)
    radius = 5
    thickness = 1

    xyxy_ce = known_parking.get('xyxy')
    max_height = 41
    for xc, yc in known_parking.get('list'):
        cv2.circle(frame, (int(xc), int(yc)), radius, (0, 255, 0), 1)
        cent_xyxy = xyxy_ce[str(xc) + str(yc)]
        print(cent_xyxy)
        width = cent_xyxy[2] - cent_xyxy[0]
        height = cent_xyxy[3] - cent_xyxy[1]
    #
    #     scaling = round(height / 41, 2)
    #     print(f"scaling {scaling}")
    #     # print(f"width {width}")
    #     # print(f"height {height}")
    #     cv2.rectangle(frame, (cent_xyxy[0], cent_xyxy[1]), (cent_xyxy[2], cent_xyxy[3]), color, 1)
    #     cv2.putText(frame, f'{int(height * scaling)}', (cent_xyxy[0], int(cent_xyxy[1] * scaling)),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
    #                 2)
    #     cv2.putText(frame, f'{int(yc)}', (cent_xyxy[0] - 20, int(cent_xyxy[1] - 10)),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    # print(max_height)
    for result in results[0].boxes:
        if result.cls in types:  # Class 0 corresponds to "person" in COCO dataset
            object_type = ObjectType.CAR.value if result.cls == ObjectType.CAR.value else ObjectType.PERSON.value
            # # Get the bounding box coordinates (x1, y1, x2, y2)
            # print(result)
            x1, y1, x2, y2 = result.xyxy[0].cpu().numpy().astype(int)
            object_coordinates = [int(x1), int(y1), int(x2), int(y2)]

            record_time = RecordsByTime(time=start_time, xyxy=object_coordinates)
            if object_type == 0:
                camera_data.people.recordsByTime.append(record_time)
            elif object_type == 2:
                camera_data.cars.recordsByTime.append(record_time)

            # founds = False
            # for parking in known_parking.get('list'):
            #     distance = euclidean((xc, yc), parking)
            #     if distance <= 15:
            #         founds = True
            #         print(f"Breaking inner loop and skipping outer iteration. {(xc, yc)}")
            #         break
            # #
            # if not founds:
            #     color = (0, 0, 255)
            #     radius = 5
            #     thickness = 1
            #     # Draw a rectangle around the person
            #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Green box, thickness 2

            # Draw a rectangle around the person
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Green box, thickness 2

            # Add a label with the word 'Person' and the count
            # cv2.putText(frame, f'{object_type}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return camera_data


def get_camera_data(camera_name, camera_id, image_url, known_parking) -> CameraData:
    frame = fetch_image(image_url)
    if frame is not None:
        camera_data = get_all_objects(frame, known_parking)
        camera_data.calculate_metrics()

        people_count = camera_data.people.calculate_objects()
        cars_count = camera_data.cars.calculate_objects()
        # cv2.imshow(f"People Counting", frame)
        # cv2.waitKey(0)
        # print(f"{people_count} persons and {cars_count} cars at {camera_name}")
        return camera_data


def run_count():
    # Loop to fetch and process the latest image every few seconds
    all_cameras = load_json_data()

    while True:
        cameras = get_all_cameras()
        all_people_count = 0

        # Format: [x1, y1, x2, y2]
        # all_cameras = []
        for camera in cameras:
            camera_name = camera['name']
            camera_id = camera['id']
            camera_url = camera['imageUrl']
            print(camera_name)
            known_parking = []
            if all_cameras and all_cameras[camera_id]:
                known_parking = all_cameras[camera_id].cars.knownParking
                if known_parking:
                    known_parking = known_parking
            camera_data = get_camera_data(camera_name, camera_id, camera_url, known_parking)
            if camera_data:
                if camera_id in all_cameras:
                    all_cameras[camera_id].people.recordsByTime.extend(camera_data.people.recordsByTime)
                    all_cameras[camera_id].cars.recordsByTime.extend(camera_data.cars.recordsByTime)
                else:
                    all_cameras[camera_id] = camera_data

            # print(image_url)
            # print(people_count)
            # all_people_count += people_count
            # cv2.destroyAllWindows()
            # Display the count of people on the image
            # cv2.putText(frame, f"People Count: {cars_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),2)

            # Show the image with bounding boxes and count

        # time.sleep(5)

        # live_stream_count = count_all_live_stream()
        # all_people_count += live_stream_count

        print("########################### people in manhattan")
        # print(all_people_count)

        save_json(all_cameras)
        cv2.destroyAllWindows()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def load_json_data():
    import json
    # Open and load the JSON file
    data = {}
    with open("data.json", "r") as file:
        try:
            data = json.load(file)
            if data:
                loaded = load_data(data)
                data = loaded.AllData if loaded and hasattr(loaded, "AllData") else {}
        except Exception as e:
            print(f"Error: {e}")
    return data


def read_json_data():
    import json
    # Open and load the JSON file
    data = {}
    with open("data.json", "r") as file:
        try:
            data = json.load(file)
        except Exception as e:
            print(f"Error: {e}")
    return data


# def detect_parking(records, time_frame):


def create_json(file_name="data.json"):
    try:
        with open(file_name, 'x') as file:
            pass  # Do nothing; this leaves the file empty
    except FileExistsError:
        print("File already exists!")


def save_json(all_data):
    with open("data.json", "w") as file:
        all_data = {key: data.to_dict() for key, data in all_data.items()}
        json.dump(all_data, file)


# live stream
def get_live_stream_url(video_url):
    command = ["yt-dlp", "-g", video_url]
    stream_url = subprocess.check_output(command).decode("utf-8").strip()
    return stream_url


def count_stream(video_url):
    try:
        # while True:
        live_stream_url = get_live_stream_url(video_url)
        print()
        print("#####")
        print("Live stream URL:", live_stream_url)

        cap = cv2.VideoCapture(live_stream_url)
        ret, frame = cap.read()

        people_count = get_all_objects(frame, [0], cof=0.01, iou=0.001)
        # cv2.imshow("People Counting with Bounding Boxes", frame)
        # cv2.waitKey(0)
        return len(people_count.people.recordsByTime)
        # cv2.putText(frame, f"People Count: {on_video}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # #
        # # # Show the image with bounding boxes and count
        #

        # print(on_video)
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord("q"):
        #     print("Exiting...")
        #     break

    except Exception as e:
        print(f"Error: {e}")


def count_all_live_stream():
    video_url = ["https://www.youtube.com/watch?v=rnXIjl_Rzy4",
                 "https://www.youtube.com/watch?v=srlpC5tmhYs",
                 "https://www.youtube.com/watch?v=VVBU63T23j8"]
    all_count = 0
    for url in video_url:
        people_count = count_stream(url)
        all_count += people_count

    print(f"All live stream count {all_count}")
    return all_count


if __name__ == "__main__":
    print("########")
    run_count()

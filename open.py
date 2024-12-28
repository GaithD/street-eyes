import cv2
import requests
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
import time
import numpy as np
import csv
import subprocess
import math

from models import ObjectType, CameraData, RecordsByTime, AllCamerasData

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


def get_all_objects(frame, types=[ObjectType.PERSON.value, ObjectType.CAR.value], cof=0.1, iou=0.01):
    """Detect and count people, and draw bounding boxes around them."""
    results = model(frame, conf=cof, iou=iou)  # Run detection on the frame
    # Loop over all detected boxes and count those that correspond to "person"
    camera_data = CameraData()
    start_time = int(time.time())
    for result in results[0].boxes:
        if result.cls in types:  # Class 0 corresponds to "person" in COCO dataset
            object_type = ObjectType.CAR.value if result.cls == ObjectType.CAR.value else ObjectType.PERSON.value
            # # Get the bounding box coordinates (x1, y1, x2, y2)
            x1, y1, x2, y2 = result.xyxy[0].cpu().numpy().astype(int)
            object_coordinates = [x1, y1, x2, y2]

            record_time = RecordsByTime(time=start_time, xyxy=object_coordinates)
            if object_type == 0:
                camera_data.people.recordsByTime.append(record_time)
            elif object_type == 2:
                camera_data.cars.recordsByTime.append(record_time)

            # Draw a rectangle around the person
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box, thickness 2

            # Add a label with the word 'Person' and the count
            cv2.putText(frame, f'{object_type}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    print(camera_data)
    return camera_data


def count_people(camera_data):
    return len(camera_data.people.recordsByTime)


def count_cars(camera_data, frame):
    # none_stationary = remove_stationary_cars(all_time_cars, all_objects[ObjectType.CAR.value], frame)
    return len(camera_data.cars.recordsByTime)


def remove_stationary_cars(all_time_cars, all_objects, frame):
    times = np.array(list(all_time_cars.keys()))
    print("timesss")
    print(times)
    time_now = time.time()
    objects_60s_old = times[time_now - times > 60]

    none_stationary = []
    for new_object in all_objects:
        new_object_times_found = 0
        for object_60s in objects_60s_old:
            if new_object_times_found > 5:
                break
            time_objects = all_time_cars.get(object_60s)
            for time_object in time_objects:
                if new_object_times_found > 5:
                    # print("exitttttttt")
                    break
                # print("new_object")
                # print(new_object)
                euclidean_dist = euclidean_distance(time_object, new_object)
                if euclidean_dist < 3:
                    new_object_times_found += 1
                    # print("#########################")
                    # print("found")
                    break
        if new_object_times_found == 0:
            x1 = new_object[0]
            y1 = new_object[1]
            x2 = new_object[2]
            y2 = new_object[3]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box, thickness 2

            # Add a label with the word 'Person' and the count
            cv2.putText(frame, f'car', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            none_stationary.append(new_object)

    print("none_stationary")
    print(none_stationary)
    return none_stationary


def run_count():
    # Loop to fetch and process the latest image every few seconds
    all_cameras = []
    while True:
        cameras = get_all_cameras()
        all_people_count = 0

        # Format: [x1, y1, x2, y2]
        # all_cameras = []
        for camera in cameras:
            image_url = camera['imageUrl']
            camera_id = camera['id']
            # image_url = 'https://webcams.nyctmc.org/api/cameras/75a7b81a-6233-47bf-8428-ea6c9edec1f8/image'
            frame = fetch_image(image_url)
            if frame is not None:
                # Count the people and draw bounding boxes

                camera_data = get_all_objects(frame)
                print("##################### Camera")
                print(camera_data)
                start_time = time.time()

                people_count = count_people(camera_data)
                cars_count = count_cars(camera_data, frame)

                all_cameras.append(AllCamerasData(id=camera_id, data=camera_data))

                print(f"{people_count} persons and {cars_count} cars at {camera['name']}")

                # print(image_url)
                # print(people_count)
                # all_people_count += people_count
                # cv2.destroyAllWindows()
                # Display the count of people on the image
                # cv2.putText(frame, f"People Count: {cars_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),2)

                # Show the image with bounding boxes and count
                # cv2.imshow(f"People Counting {camera['area']} ", frame)
                # cv2.waitKey(0)

            # Wait a bit before fetching the next image (adjust as needed)

            # Break on 'q' key press
        # time.sleep(5)
        print("###########ALL")

        merged_dict = {obj.id: obj.to_dict() for obj in all_cameras}
        print(merged_dict)
        live_stream_count = count_all_live_stream()
        all_people_count += live_stream_count

        print("########################### people in manhattan")
        print(all_people_count)
        save_log(all_people_count)
        cv2.destroyAllWindows()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def save_log(all_people_count):
    # Data to append
    start_time = time.time()
    data = [[start_time, all_people_count]]

    # Open the CSV file in append mode
    with open('data.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Append the data
        writer.writerows(data)


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

import json

from open import read_json_data
from scipy.spatial.distance import euclidean
from itertools import combinations
from collections import defaultdict
import pandas as pd


def load_df(data):
    return pd.DataFrame(data)


def calculate_centroid(xyxy):
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2, (y1 + y2) / 2


def calculate_distances(centroids, threshold=3):
    pairs_below_threshold = []
    for c1, c2 in combinations(centroids, 2):
        distance = euclidean(c1, c2)
        if distance <= threshold:
            pairs_below_threshold.extend((c1, c2))
    return pairs_below_threshold


def group_centroid_by_time_period(df, period):
    df['3_min_period'] = (df['time'] // period) * period
    df['centroid'] = df['xyxy'].apply(calculate_centroid)

    centroid_xyxy = {}
    for _, row in df.iterrows():
        c1, c2 = row['centroid']
        centroid_xyxy[str(c1) + str(c2)] = row['xyxy']

    grouped = df.groupby('3_min_period').agg(
        centroids=('centroid', list)
        # List of centroids in each group
    ).reset_index()

    grouped['distances_below_threshold'] = grouped['centroids'].apply(lambda x: calculate_distances(x, threshold=3))

    return grouped, centroid_xyxy


def detect_parking(camera):
    cars = camera.get("cars").get("recordsByTime")
    if cars:
        df = load_df(cars)

        group, centroid_xyxy = group_centroid_by_time_period(df, 300)
        return group, centroid_xyxy
    return None, None


def detect_all_parking():
    data = read_json_data()
    print(data)
    for camera_id in data:
        print(camera_id)
        camera = data.get(camera_id)
        group, centroid_xyxy = detect_parking(camera)
        if isinstance(group, pd.DataFrame):
            scans_count = len(group)
            print(f"scan total {scans_count}")
            distance_threshold = 3

            unique_centroids = set()

            clusters = defaultdict(list)

            group_arr = []
            for count, spot in group['distances_below_threshold'].items():
                group_arr.extend(spot)

            for c1 in group_arr:
                found_cluster = False
                for cluster_key in clusters.keys():
                    if euclidean(c1, cluster_key) <= distance_threshold:
                        clusters[cluster_key].append(c1)
                        found_cluster = True
                        break
                if not found_cluster:
                    clusters[c1].append(c1)  # Create a new cluster

            cluster_counts = {cluster: len(points) for cluster, points in clusters.items()}
            min_occurrences = max(int(scans_count * 0.50), 40)
            print(f"min_occurrences {min_occurrences}")
            possible_parking_spots = {cluster: count for cluster, count in cluster_counts.items() if
                                      count >= min_occurrences}

            parkings_final = []
            final_xyxy = {}
            for spot, count in possible_parking_spots.items():
                parkings_final.append(spot)
                final_xyxy[str(spot[0]) + str(spot[1])] = centroid_xyxy[str(spot[0]) + str(spot[1])]

            data[camera_id]["cars"]["knownParking"] = {"list": parkings_final, "xyxy": final_xyxy}

    return data


def save_json(all_data):
    with open("data.json", "w") as file:
        json.dump(all_data, file)


if __name__ == "__main__":
    print("########")
    data = detect_all_parking()
    save_json(data)

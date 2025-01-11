from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
import json
from dacite import from_dict


class ObjectType(Enum):
    PERSON = 0
    CAR = 2


# @dataclass
# class AllTimeMetrics:
#     max: int = 0
#     min: int = 0
#     average: int = 0
#     totalCount: int = 0


@dataclass
class RecordsByTime:
    time: int
    xyxy: list[int] = field(default_factory=list)


@dataclass
class CommonData:
    # allTime: Optional[AllTimeMetrics] = field(default_factory=AllTimeMetrics)
    recordsByTime: List[RecordsByTime] = field(default_factory=list[RecordsByTime])

    def calculate_objects(self) -> int:
        return len(self.recordsByTime)


@dataclass
class CarsData(CommonData):
    knownParking: dict = field(default_factory=dict)


@dataclass
class CameraData:
    cars: CarsData = field(default_factory=CarsData)
    people: CommonData = field(default_factory=CommonData)

    def load(self):
        self.cars = CommonData()

    def calculate_metrics(self):
        return self.cars.calculate_objects() + self.people.calculate_objects()

    def to_dict(self) -> Dict:
        # Convert SubObject to a dictionary for serialization
        return asdict(self)


@dataclass
class AllCamerasData:
    AllData: Dict[str, CameraData]

    # def to_dict(self) -> Dict:
    #     # Convert SubObject to a dictionary for serialization
    #     return asdict(self)
    #
    # def to_json(self) -> str:
    #     # Serialize the JsonResponse to JSON format
    #     return json.dumps({
    #         self.id: self.data.to_dict()  # Map id to the SubObject as a dictionary
    #     }, indent=4)


# Function to load data
def load_data(json_data: str) -> AllCamerasData:
    # Parse JSON string to dictionary
    raw_data = json_data

    # Convert dictionary values to MyObject instances

    objects = {key: from_dict(data_class=CameraData, data=value) for key, value in raw_data.items()}

    # Return the dataclass
    return AllCamerasData(AllData=objects)


# Example usage
if __name__ == "__main__":
    all_time_metrics = AllTimeMetrics(max=100, min=10, average=55.5, totalCount=200)
    records = [RecordsByTime(time=1672531200, xyxy=["134", "3434", "534", "347"])]
    common_data_people = CommonData(allTime=all_time_metrics, recordsByTime=records)
    cars_data = CarsData(
        allTime=all_time_metrics,
        recordsByTime=records,
        knownParking={"Lot A": {"count": 10}}
    )
    camera_data = CameraData(cars=cars_data, people=cars_data)
    all_cameras_data = AllCamerasData(id="345345-34434", data=camera_data)

    print(all_cameras_data.data.calculate_metrics())

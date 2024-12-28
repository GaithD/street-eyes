from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
import json


class ObjectType(Enum):
    PERSON = 0
    CAR = 2


@dataclass
class AllTimeMetrics:
    max: int = 0
    min: int = 0
    average: int = 0
    totalCount: int = 0


@dataclass
class RecordsByTime:
    time: int
    xyxy: list[str] = field(default_factory=list)


@dataclass
class CommonData:
    allTime: Optional[AllTimeMetrics] = field(default_factory=AllTimeMetrics)
    recordsByTime: List[RecordsByTime] = field(default_factory=list)


@dataclass
class CarsData(CommonData):
    knownParking: dict = field(default_factory=dict)


@dataclass
class CameraData:
    cars: CarsData = field(default_factory=CarsData)
    people: CommonData = field(default_factory=CommonData)

    def to_dict(self) -> Dict:
        # Convert SubObject to a dictionary for serialization
        return asdict(self)


@dataclass
class AllCamerasData:
    id: str
    data: CameraData

    def to_dict(self) -> Dict:
        # Convert SubObject to a dictionary for serialization
        return asdict(self)

    def to_json(self) -> str:
        # Serialize the JsonResponse to JSON format
        return json.dumps({
            self.id: self.data.to_dict()  # Map id to the SubObject as a dictionary
        }, indent=4)


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
    camera_data = CameraData()
    all_cameras_data = AllCamerasData(id="345345-34434", data=camera_data)

    print(all_cameras_data.to_json())

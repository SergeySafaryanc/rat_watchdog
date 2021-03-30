import json
from typing import List, Any

from pydantic import BaseModel


class Config(BaseModel):
    HOST: str
    PORT: int
    wait_time: int
    epoch_time: float
    sampling_rate: int
    decimate_rate: int
    num_channels: int
    num_counter_for_refresh_animal: int
    count_train_stimuls: int
    train_step: int
    data_source_is_file: bool
    is_result_validation: bool
    is_train: bool
    use_auto_train: bool

    odors: List[List[str]]
    odors_set: List[int]
    weights: List[float]
    unite: List[List[int]]
    unite_test: List[List[int]]

    rat_name: str = ""

    def __init__(self, file: str, is_refresh: bool = False, **data: Any) -> None:
        super().__init__(**json.load(open(file, encoding="UTF-8")))
        if is_refresh:
            super(Config, self).__init__(**data)
            json.dump(self.dict(), open(file, "w"), ensure_ascii=False, indent=4)



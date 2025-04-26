from pydantic import BaseModel
from typing import List, Union
from enum import Enum

class UserInfo(BaseModel):
    weight: Union[int, float]
    height: Union[int, float]
    bmi: Union[int, float]
    body_fat_percentage: Union[int, float]
    gender: str
    age: int

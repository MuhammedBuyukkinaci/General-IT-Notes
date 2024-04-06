from enum import Enum
from typing import List, Optional, Set, Union

from fastapi import (Body, Cookie, Depends, FastAPI, File, Form, HTTPException,
                     Path, Query, UploadFile, status)
from pydantic import BaseModel


# A pydantic class for post method
class Item(BaseModel):
    name: str
    price: float
    brand: Optional[str] = None


# A pydantic class for put method
class UpdateItem(BaseModel):
    name: Optional[str] = None
    price: Optional[float] = None
    brand: Optional[str] = None


# Define some boilerplate data
inventory = {
    1: {"name": "bread", "price": 5.99},
    2: {"name": "milk", "price": 20.99},
    3: {"name": "egg", "price": 3.0},
}

# Creating a FastAPI instance
app = FastAPI()


@app.get("/")
def home():
    return {"Data": "HomePage"}


@app.get("/about")
def home():
    return {"Data": "About"}


@app.get("/get-item/{item_id}")
def get_item(item_id: int):
    return inventory[item_id]


@app.get("/get-item/{item_id}/{name}")
def get_item_2_path_params(
    item_id: int = Path(None, description="The description to show in docs", gt=1.0)
):
    return inventory[item_id]


# the below query parameter is obligaroty and should be filled out.
@app.get("/get-by-name")
def get_item_by_query(name: str):
    for item_id in inventory:
        if inventory[item_id]["name"] == name:
            return inventory[item_id]
    return HTTPException(status_code=404, detail="Item name not found")


# None makes this query parameter optional.
@app.get("/get-by-name-optional")
def get_item_by_query_optional(name: str = None):
    for item_id in inventory:
        if inventory[item_id]["name"] == name:
            return inventory[item_id]
    return {"Data": "Not Found"}


# None makes this query parameter optional.
@app.get("/get-by-name-path-param-and-query/{item_id}")
def get_item_by_path_param_query_param(item_id: int, test: int, name: str = None):
    if item_id in inventory:
        return inventory[item_id]
    return {"Data": "Not Found"}


## Post
inventory_new = {}


@app.post("/create-item/{item_id}")
def create_item(item_id: int, item: Item):
    if item_id in inventory_new:
        # 400 means bad request
        return HTTPException(status_code=400, detail="Item already exists")
    inventory_new[item_id] = item
    return inventory_new[item_id]


## Put
@app.put("/update-item/{item_id}")
def update_item(item_id: int, item: UpdateItem):
    if item_id not in inventory_new:
        return HTTPException(status_code=404, detail="Item ID not found")

    if item.name is not None:
        inventory_new[item_id].name = item.name
    if item.price is not None:
        inventory_new[item_id].price = item.price
    if item.brand is not None:
        inventory_new[item_id].brand = item.brand
    return inventory_new[item_id]


## Delete
@app.delete("/delete-item")
def delete_item(
    item_id: int = Query(..., description="The ID of the item to delete", gt=0)
):
    if item_id not in inventory_new:
        return HTTPException(status_code=404, detail="Item name not found to delete")
    del inventory_new[item_id]
    return {"Success": "Item deleted"}


# Enum & str

from enum import Enum


class Sport(str, Enum):
    football = "football"
    basketball = "basketball"


@app.get("/get-sport/{sport}")
def get_sport(sport: Sport):
    if sport == Sport.basketball:
        return {"Sport": "basketball"}
    # another way of accessing value
    if sport.value == "football":
        return {"Sport": "football"}
    return {"Message": "Sport not found in Enum class"}


# Post and BaseModel


class MlModel(BaseModel):
    name: str
    number_of_param: int
    size_of_model: str = None


@app.post("/post-model/")
def post_ml_model(ml_model: MlModel):
    return ml_model.name, ml_model.number_of_param, ml_model.size_of_model


@app.get("/get-multiple-query/")
def get_multiple_query(q: Union[List[str], None] = Query(default=None)):
    return q


# Field Example
from pydantic import Field


class ItRole(BaseModel):
    role: Union[str, None] = Field(default=None, max_length=10)
    writes_code: bool = Field(
        title="whether the role writes code or not",
        description="True if the rol writes code else False",
    )


@app.post("/it-roles/{role_number}")
async def update_item(role_number: int, it_role: ItRole = Body(embed=True)):
    results = {"item_id": role_number, "it_role": it_role}
    return results


# List & Set fields


class ComputerPieces(BaseModel):
    is_collected: bool  # it is obligatory, can't be None
    unique_pieces: Set[str]  # it is obligatory, can't be None
    all_pieces: List[str]  # it is obligatory, can't be None


@app.post("/computer-pieces/")
async def post_computer_pieces(computer_piece: ComputerPieces):
    results = {"computer_piece": computer_piece}
    print(results)
    return results


# Nested Models
class Teacher(BaseModel):
    name: str
    specialty: str


class Student(BaseModel):
    age: int
    teacher: Teacher


@app.post("/post-student/")
async def post_student(student: Student):
    print(student)
    return student


# Providing example data to be displayed in documentations
class Job(BaseModel):
    title: str
    salary: float

    class Config:
        schema_extra = {"example": {"title": "Developer", "salary": 10000.5}}


@app.post("/post-job/")
async def post_job(job: Job):
    results = {"job": job}
    return results


class Expertise(BaseModel):
    title: str = Field(example="A very nice Item")
    salary: float = Field(example=10000.5)


@app.post("/post-expertise/")
async def post_expertise(expertise: Expertise):
    results = {"expertise": expertise}
    return results


## Response Model for decorators


class Listing(BaseModel):
    listing_id: str
    price: int


@app.get("/response-model-example/", response_model=List[Listing])
def get_listings():
    return_element = [
        {"listing_id": "012", "price": 10},
        {"listing_id": "013", "price": 11},
    ]
    return return_element


## Response model that is able to return 2 different classes


class Car(BaseModel):
    name: str
    speed: int


class Plane(BaseModel):
    name: str
    speed: int
    is_expensive: bool


vehicles = {
    "vehicle1": {"name": "BMW", "speed": 200},
    "vehicle2": {"name": "Boeing", "speed": 1000, "is_expensive": True},
}


@app.get("/get-vehicle/{vehicle_name}", response_model=Union[Car, Plane])
def get_vehicle(vehicle_name: str):
    return vehicles[vehicle_name]


## Form to receive sensitive info


@app.post("/post-via-form/")
def post_via_form(username: str = Form(), password: str = Form()):
    return (username, password)


## File && UploadFile


@app.post("/post-file/")
def post_file(file: bytes = File()):
    return {"file_size": len(file)}


@app.post("/post-upload-file/")
def post_upload_file(upload_file: UploadFile):
    return {"filename": upload_file.filename}


## HttpException

products_dict = {"egg": 0.99}


@app.get("/get-http-example/{product_name}")
def get_http_exception_example(product_name: str):
    if product_name not in products_dict:
        raise HTTPException(status_code=404, detail="product not found")
    return products_dict[product_name]


# Path Operation Configuration
## Multiple tags possbile thanks to using list.

class Computer(BaseModel):
    name: str


class TagEnum(Enum):
    computers = "computers"


@app.post(
"/get-computer/",
tags=[TagEnum.computers],
description="description is here", 
summary="summary is here",
response_description="The response is here",
deprecated=False,
response_model=Computer)
def post_computer(computer_name: Computer):
    """
    Create an item with all the information:

    - **name**: the name of the computer
    """
    return computer_name

## Dependencies

async def common_parameters(q: Union[str, None] = None):
    return {"q": q}

@app.get("/dependency-injection-1/")
def get_dependency_injection(commons: dict = Depends(common_parameters)):
    return commons

@app.get("/dependency-injection-2/")
def get_dependency_injection(commons: dict = Depends(common_parameters)):
    return commons


from typing import Any

from pydantic import (EmailStr, Field, SecretStr, ValidationError,
                      field_validator, model_validator)


class Student(BaseModel):
    num: int = Field(examples=[1,2],
                     description='Indicating the number of a student',
                     frozen=True)
    # Frozen means whether it is allowed to be modified or not
    # If frozen is set to True, it can be changed. If frozen is set
    # to False, it can't be set.
    email: EmailStr = Field(examples=['muhammed@gmail.com'],description='Indicating email',
                                      frozen=False)
    # exclude = True means when the object is serialized, password field is excluded
    password: SecretStr = Field(examples=['123456'],exclude=True)

def validation(inp: Any):
    try:
        student = Student.model_validate(inp)
        print(student)
    except ValidationError as e:
        print("student in not valid")
        for error in e.errors():
            print(error)










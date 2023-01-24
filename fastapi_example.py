from typing import List, Optional, Union

from fastapi import FastAPI, HTTPException, Path, Query, status
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
@app.put('/update-item/{item_id}')
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
@app.delete('/delete-item')
def delete_item(item_id: int = Query(..., description = "The ID of the item to delete",gt=0)):
    if item_id not in inventory_new:
        return HTTPException(status_code=404, detail="Item name not found to delete")
    del inventory_new[item_id]
    return {"Success": "Item deleted"}

# Enum & str

from enum import Enum


class Sport(str, Enum):
    football = "football"
    basketball = "basketball"


@app.get('/get-sport/{sport}')
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


@app.post('/post-model/')
def post_ml_model(ml_model: MlModel):
    return ml_model.name, ml_model.number_of_param, ml_model.size_of_model

@app.get('/get-multiple-query/')
def get_multiple_query(q: Union[List[str], None]= Query(default=None)):
    return q




from typing import Optional

from fastapi import FastAPI, Path
from pydantic import BaseModel


class Item(BaseModel):
    name: str
    price: float
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

@app.get("/get-by-name")
def get_item_by_query(name: str):
    for item_id in inventory:
        if inventory[item_id]["name"] == name:
            return inventory[item_id]
    return {"Data": "Not Found"}

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
        return {"Error": "Item already exists "}
    inventory_new[item_id] = item
    return inventory_new[item_id]

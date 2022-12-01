import random
import json


OBJECTS = [
    ## '''stationary'''
    # "table",
    # "sink",
    # "dish_rack",
    # "oven",

    #'''bowls'''
    "gray_bowl",
    "black_bowl",
    "blue_bowl",

    #'''plates'''
    "white_plate",
    "gray_plate",

    #'''cups'''
    "green_cup",
    "yellow_cup",

    #'''breads'''
    "long_bread",
    "square_bread",

    #'''dairy'''
    "milk_dairy",
    "butter_dairy",

    #'''meats'''
    "burger_meat",
    "steak_meat",

    #'''fruits'''
    "apple_fruit",
    "orange_fruit"
]

def read_json(path: str):
    '''Read JSON file into dictionary.

    Args:
        path [str]: path to the JSON file
    Returns:
        data [Any]: data of JSON file
    '''
    with open(path, "r") as F:
        return json.load(F)


def write_json(obj, dest_path: str):
    '''Write JSON file to destination path.

    Args:
        obj [Any]: data to be written to JSON
        dest_path [str]: destination path of the JSON file
    '''
    with open(dest_path, "w") as F:
        json.dump(obj, F)


if __name__ == "__main__":
    records_path = "samples_records.json"
    min_items = 4
    max_items = 6

    records = read_json(records_path)
    num_items = random.randint(min_items, max_items)
    items = random.sample(OBJECTS, num_items)
    print(items)
    accept = input("accept?: ")
    if accept == "y":
        for item in items:
            records[item] += 1
        write_json(records, records_path)

    # '''reset json'''
    # records = {}
    # for obj in OBJECTS:
    #     records[obj] = 0
    # write_json(records, records_path)

import random
import json

from databoost.utils.data import read_json, write_json


OBJECTS = [
    # '''stationary'''
    # "table",
    # "sink",
    # "dish_rack",
    # "oven",

    # '''bowls'''
    "gray_bowl",
    "black_bowl",
    "blue_bowl",

    # '''plates'''
    "white_plate",
    "gray_plate",

    # '''cups'''
    "green_cup",
    "yellow_cup",

    # '''breads'''
    "long_bread",
    "square_bread",

    # '''dairy'''
    "milk_dairy",
    "butter_dairy",

    # '''meats'''
    "burger_meat",
    "steak_meat",

    # '''fruits'''
    "apple_fruit",
    "orange_fruit"
]


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

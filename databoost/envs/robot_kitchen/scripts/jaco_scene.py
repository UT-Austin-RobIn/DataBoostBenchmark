import copy
import random
import time
from typing import Set


class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattr__(self, attr):
        # Take care that getattr() raises AttributeError, not KeyError.
        # Required e.g. for hasattr(), deepcopy and OrderedDict.
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError("Attribute %r not found" % attr)

    def __getstate__(self):
        return self

    def __setstate__(self, d):
        self = d


# OBJECTS contain object names as keys and information about the object as
# its values, such as specifications of what actions are valid for a given
# object and which receptacles it can be put into
OBJECTS = AttrDict({
    "gray_bowl": AttrDict({
        "class": "bowl",
        "actions": ["go_to", "slide_to", "pick", "place", "switch"],
        "receptacles": ["table", "sink", "dish_rack", "oven", "plate"]
    }),
    "black_bowl": AttrDict({
        "class": "bowl",
        "actions": ["go_to", "slide_to", "pick", "place", "switch"],
        "receptacles": ["table", "sink", "dish_rack", "oven", "plate"]
    }),
    "blue_bowl": AttrDict({
        "class": "bowl",
        "actions": ["go_to", "slide_to", "place", "switch"]
    }),
    "red_bowl": AttrDict({
        "class": "bowl",
        "actions": ["go_to", "slide_to", "place", "switch"]
    }),
    "white_plate": AttrDict({
        "class": "plate",
        "actions": ["go_to", "place", "switch"]
    }),
    "gray_plate": AttrDict({
        "class": "plate",
        "actions": ["go_to", "slide_to", "place", "switch"]
    }),
    "red_cup": AttrDict({
        "class": "cup",
        "actions": ["go_to", "slide_to", "switch"]
    }),
    "green_cup": AttrDict({
        "class": "cup",
        "actions": ["go_to", "slide_to", "pick", "switch"],
        "receptacles": ["table", "sink", "dish_rack", "oven", "plate"]
    }),
    "yellow_cup": AttrDict({
        "class": "cup",
        "actions": ["go_to", "slide_to", "pick", "switch"],
        "receptacles": ["table", "sink", "dish_rack", "oven", "plate"]
    }),
    "round_bread": AttrDict({
        "class": "bread",
        "actions": ["go_to", "slide_to", "pick", "switch"],
        "receptacles": ["table", "sink", "oven", "plate", "box", "bowl"]
    }),
    "long_bread": AttrDict({
        "class": "bread",
        "actions": ["go_to", "slide_to", "pick", "switch"],
        "receptacles": ["table", "sink", "oven", "plate", "box", "bowl"]
    }),
    "square_bread": AttrDict({
        "class": "bread",
        "actions": ["go_to", "slide_to", "pick", "switch"],
        "receptacles": ["table", "sink", "oven", "plate", "box", "bowl"]
    }),
    "steak_meat": AttrDict({
        "class": "meat",
        "actions": ["go_to", "slide_to", "pick", "switch"],
        "receptacles": ["table", "sink", "oven", "plate", "box", "bowl"]
    }),
    "burger_meat": AttrDict({
        "class": "meat",
        "actions": ["go_to", "slide_to", "pick", "switch"],
        "receptacles": ["table", "sink", "oven", "plate", "box", "bowl"]
    }),
    "drumstick_meat": AttrDict({
        "class": "meat",
        "actions": ["go_to", "slide_to", "pick", "switch"],
        "receptacles": ["table", "sink", "oven", "plate", "box", "bowl"]
    }),
    "fish_meat": AttrDict({
        "class": "meat",
        "actions": ["go_to", "slide_to", "pick", "switch"],
        "receptacles": ["table", "sink", "oven", "plate", "box", "bowl"]
    }),
    "milk_dairy": AttrDict({
        "class": "dairy",
        "actions": ["go_to", "slide_to", "pick", "switch"],
        "receptacles": ["table", "sink", "plate", "box", "bowl"]
    }),
    "cheese_dairy": AttrDict({
        "class": "dairy",
        "actions": ["go_to", "slide_to", "pick", "switch"],
        "receptacles": ["table", "sink", "plate", "box", "bowl"]
    }),
    "butter_dairy": AttrDict({
        "class": "dairy",
        "actions": ["go_to", "slide_to", "pick", "switch"],
        "receptacles": ["table", "sink", "plate", "box", "bowl"]
    }),
    "apple_fruit": AttrDict({
        "class": "fruit",
        "actions": ["go_to", "slide_to", "pick", "switch"],
        "receptacles": ["table", "sink", "dish_rack", "oven", "plate", "box", "bowl"]
    }),
    "banana_fruit": AttrDict({
        "class": "fruit",
        "actions": ["go_to", "slide_to", "pick", "switch"],
        "receptacles": ["table", "sink", "dish_rack", "oven", "plate", "box", "bowl"]
    }),
    "watermelon_fruit": AttrDict({
        "class": "fruit",
        "actions": ["go_to", "slide_to", "pick", "switch"],
        "receptacles": ["table", "sink", "dish_rack", "oven", "plate", "box", "bowl"]
    }),
    "white_fruit": AttrDict({
        "class": "fruit",
        "actions": ["go_to", "slide_to", "pick", "switch"],
        "receptacles": ["table", "sink", "dish_rack", "oven", "plate", "box", "bowl"]
    }),
    "box": AttrDict({
        "class": "box",
        "actions": ["go_to", "slide_to", "pick", "place", "switch"],
        "receptacles": ["table", "sink", "dish_rack"]
    }),
    "sink": AttrDict({
        "class": "sink",
        "actions": ["go_to", "place"]
    }),
    "dish_rack": AttrDict({
        "class": "dish_rack",
        "actions": ["go_to", "place"]
    }),
    "oven": AttrDict({
        "class": "oven",
        "actions": ["go_to", "place"]
    }),
    "table": AttrDict({
        "class": "table",
        "actions": ["place"]
    }),
})


# Helpful reformulations of the OBJECTS data
OBJECTS_LIST = set(OBJECTS.keys())
ACTIONS_LIST = set()
CLASS_OBJECT_MAP = AttrDict()
ACTION_OBJECT_MAP = AttrDict()
OBJECT_RECEPTACLE_MAP = AttrDict()

'''This block builds the aforementioned helpful reformulations'''
for obj_name, obj_info in OBJECTS.items():
    obj_class = obj_info["class"]
    if obj_class not in CLASS_OBJECT_MAP:
        CLASS_OBJECT_MAP[obj_class] = set()
    CLASS_OBJECT_MAP[obj_class].add(obj_name)
    for action in obj_info.actions:
        ACTIONS_LIST.add(action)
        if action not in ACTION_OBJECT_MAP:
            ACTION_OBJECT_MAP[action] = set()
        ACTION_OBJECT_MAP[action].add(obj_name)

for obj_name, obj_info in OBJECTS.items():
    if "receptacles" not in obj_info:
        continue
    OBJECT_RECEPTACLE_MAP[obj_name] = set()
    for receptacle in obj_info.receptacles:
        OBJECT_RECEPTACLE_MAP[obj_name] = OBJECT_RECEPTACLE_MAP[obj_name].union(
            CLASS_OBJECT_MAP[receptacle])


class JacoScene:
    def __init__(self, in_scene = Set[str], switch_out_freq: int = 5):
        '''JacoScene keeps track of the current scene state and contains
        logic of what actions can be performed at any given time.

        Args:
            in_scene [Set[str]]: list of objects to initialize the scene.
                                  objects passed in must be from the
                                  OBJECTS dictionary
            switch_out_freq [int]: period at which a "switch" prompt is given.
                                   If a switch is not feasible, it will be
                                   skipped
        Attributes:
            switch_out_freq [int]: period at which a "switch" prompt is given.
                                   If a switch is not feasible, it will be
                                   skipped
            in_scene [Set[str]]: list of objects currently in the scene
            outside_scene [Set[str]]: list of objects not currently in the scene
            filled_receptacles [Dict[str, Set[str]]]: dictionary where keys are
                                                      filled receptacles, and
                                                      their values are the objects
                                                      they contain
            in_hand [str]: the object currently in the robot's hand; if empty, None
        '''
        in_scene = set(in_scene)  # deduplication
        for obj in in_scene:
            assert obj in OBJECTS_LIST, f"obj {obj} not a valid obj"
        self.switch_out_freq = switch_out_freq
        self.in_scene = in_scene
        self.outside_scene = OBJECTS_LIST - self.in_scene
        self.filled_receptacles = {}
        self.in_hand = None

    def prompt(self):
        '''Generates feasible prompts of action-object pairs

        Returns:
            action-object pair [Tuple[str, Tuple[str]]]: a tuple of action and
                                                         a list of corresponding
                                                         objects
        '''
        step_count = 0
        while(True):
            try:
                step_count += 1
                # At given interval, perform a "switch" prompt, if feasible
                if step_count % self.switch_out_freq == 0 and len(self.outside_scene) > 0:
                    action = "switch"
                    # determine objects that are currently in receptacles;
                    # those will not be switched out
                    in_receptacle = set([
                        obj for obj in self.filled_receptacles.values()])
                    # determine list of removable objects from the scene
                    removable_objects = self.in_scene & ACTION_OBJECT_MAP["switch"] - in_receptacle
                    # if none can be removed, skip this switch and try again
                    # next interval
                    if len(removable_objects) <= 0:
                        continue
                    # determine objects that can be added to the scene
                    addable_objects = self.outside_scene & ACTION_OBJECT_MAP["switch"]

                    # select an object to remove and an object to add
                    obj_to_remove = random.choice(list(removable_objects))
                    obj_to_add = random.choice(list(addable_objects))

                    yield("switch", (obj_to_remove, obj_to_add))

                    # update the tracking of in-scene & out-of-scene objects
                    self.in_scene.remove(obj_to_remove)
                    self.in_scene.add(obj_to_add)
                    self.outside_scene.remove(obj_to_add)
                    self.outside_scene.add(obj_to_remove)
                    continue

                # check if robot is holding something
                if self.in_hand is not None:
                    # if currently holding something, have a 50% chance of selecting
                    # "place" as next action; and no chance of "pick"
                    action = random.choice(
                        list(ACTIONS_LIST - set(["pick", "place", "switch"])))
                    action = random.choice([action, "place"])
                else:
                    # if not holding anything, no chance of "place"
                    action = random.choice(
                        list(ACTIONS_LIST - set(["place", "switch"])))
                
                # shortlist actionable objects given the chosen action
                possible_objects = self.in_scene & ACTION_OBJECT_MAP[action]
                if action == "place":
                    possible_objects = possible_objects & \
                                       OBJECT_RECEPTACLE_MAP[self.in_hand] - \
                                       set(self.filled_receptacles.keys())

                # choose object(s) to interact with. "slide_to" requires
                # 2 objects (object to slide, and destination)
                if action == "slide_to":
                    obj = random.sample(possible_objects, 2)
                else:
                    obj = random.sample(possible_objects, 1)

                yield(action, obj)

                # update tracking of in_hand and filled_receptacles
                if action == "pick":
                    obj = obj[0]  # unpack from tuple form
                    self.in_hand = obj
                    for recep, obj in self.filled_receptacles.items():
                        if obj == self.in_hand:
                            del self.filled_receptacles[recep]
                            break
                elif action == "place":
                    obj = obj[0]  # unpack from tuple form
                    if obj != "table":
                        self.filled_receptacles[obj] = self.in_hand
                    self.in_hand = None
            except Exception as e:
                print(e)
                print("---Action---")
                print(action)
                print("---Possible Objects---")
                print(possible_objects)
                print("---Actionable Items---")
                print(ACTION_OBJECT_MAP[action])
                print("---In Scene---")
                print(self.in_scene)
                print("---Outside Scene---")
                print(self.outside_scene)
                print("---Filled Receptacles---")
                print(self.filled_receptacles)
                print("---In Hand---")
                print(self.in_hand)
                if action == "place":
                    print("---Receptacles---")
                    print(OBJECT_RECEPTACLE_MAP[self.in_hand])
                return


if __name__ == "__main__":
    in_scene = [
        "table",
        "sink",
        "dish_rack",
        "oven",
        "gray_bowl",
        "white_plate",
        "green_cup",
        "long_bread",
        "burger_meat",
        "butter_dairy",
        "banana_fruit"
    ]
    scene = JacoScene(in_scene=in_scene, switch_out_freq=10)
    for action, obj in scene.prompt():
        print(f"action: {action}, object: {obj}")
        # time.sleep(1)
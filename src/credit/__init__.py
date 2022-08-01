import json
import os
import re
import time


# --------------------------------------------------------------------------------
# Global Variables
# --------------------------------------------------------------------------------
PATH_DATA = "data"
PATH_OUTPUT = "output"

SEED_TRAIN_TEST_SPLIT = 42
SEED_RANDOM_OVER_SAMPLER = 42
SEED_MODEL = 42


# --------------------------------------------------------------------------------
# Private Variables
# --------------------------------------------------------------------------------
__FILENAME__ = "src/config.json"
__RESET_GLOBALS__ = True


# --------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------
def save_globals(filename=None):
    if filename is None:
        filename = __FILENAME__
    
    variables = {
        key:value for key, value in globals().items()
        if re.match(r"(?!__)[A-Z0-9_]+(?!__)", key)
    }
    
    for _ in range(10):
        try:
            with open(filename, 'w') as f:
                json.dump(variables, f, indent=4)
            
            break
        except PermissionError as e:
            time.sleep(0.5)
    else:
        raise e


def load_globals(filename=None):
    if filename is None:
        filename = __FILENAME__
        
    for _ in range(10):
        try:
            with open(filename, 'r') as f:
                params = json.loads(f.read())
            
            break
        except PermissionError as e:
            time.sleep(0.5)
    else:
        raise e
        
    for key, value in params.items():
        if key in globals().keys():
            globals()[key] = value


def get_global(key:str):
    return globals()[key]


def set_global(key:str, value):
    if key not in globals().keys():
        raise KeyError(f"The name '{key}' is an invalid global variable")
    
    globals()[key] = value
    
    save_globals()


# --------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------
if __RESET_GLOBALS__ or not os.path.exists(__FILENAME__):
    save_globals()
else:
    load_globals()
    
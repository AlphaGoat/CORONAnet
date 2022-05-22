"""
General utilities for CORONAnet module that fit nowhere else 

Author: Peter Thomas 
Date: 17 April 2022 
"""
import os
import json


def ask_for_confirmation(prompt):
    while True:
        confirmation = input(prompt + " (Y/n): ")
        if confirmation.lower() == 'yes' or confirmation.lower() == 'y':
            return True 
        elif confirmation.lower() == 'no' or confirmation.lower() == 'n':
            return False 
        else:
            print(f"Response '{confirmation}' not understood. Asking again...")


def get_basename(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]


def is_jsonable(x):
    """
    Check that input is json serializable
    """
    try:
        json.dumps(x)
        return True
    except:
        return False

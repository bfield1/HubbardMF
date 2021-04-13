#!/usr/bin/python3

"""
Simple progress bar.

Carriage return doesn't work in IDLE, so I'm making a
custom progress bar which doesn't rely on it.
Assumes that no other printing takes place.
Author: Bernard Field
Last Modified: 2020-08-11
    Copyright (C) 2021 Bernard Field, GNU GPL v3+
"""

import sys

# Global variables.
initialised=False

def new(charlen,new_maxval,new_minval=0):
    """
    Initialise a progress bar.
    
    Inputs: charlen - length of progress bar in characters.
        new_maxval - maximum value the variable will take.
        new_minval - minimum value the variable will take.
    """
    # Invoke globals
    global minval
    global maxval
    global current_val
    global bars
    global maxbars
    global initialised
    # Print the header of the progress bar.
    print("|" + "-"*charlen + "|")
    print("|",end='') # Print the start of the progress bar.
    sys.stdout.flush() # Some terminals don't show until they get
    # a newline
    # Initialise things.
    minval = new_minval # Minimum value.
    maxval = new_maxval # Maximum value.
    current_val = new_minval
    bars = 0 # Number of bars currently printed.
    maxbars = charlen # Maximum number of bars to print.
    initialised = True

def update(newval):
    """
    Update the progress bar, possibly printing more dots.
    
    Inputs: newval - new value for the progress bar.
    Increases the progress bar appropriately.
    """
    # Invoke globals
    global minval
    global maxval
    global current_val
    global bars
    global maxbars
    global initialised
    # If we have not created a progress bar, do nothing.
    if not initialised:
        return
    # If we have already reached the end, don't do anything.
    if current_val >= maxval:
        return
    # Determine if we need to print a new bar.
    current_val = newval
    newbars = int((newval-minval)/(maxval-minval)*maxbars)
    while newbars > bars:
        bars += 1
        print("*",end='')
        sys.stdout.flush()
    # have we reached the end? If so, stop.
    if newval >= maxval:
        print("|")

def end():
    """
    Ends the progress bar.
    """
    global maxval
    update(maxval)

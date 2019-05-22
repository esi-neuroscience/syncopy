# -*- coding: utf-8 -*-
#
# Auxiliary functions for querying things/people
# 
# Created: 2019-05-22 15:40:19
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-05-22 16:19:41>

__all__ = []


def user_yesno(msg, default=None):
    """
    Docstring
    """

    # Parse optional `default` answer
    valid = {"yes": True, "y": True, "ye":True, "no":False, "n":False}
    if default is None:
        suffix = " [y/n] "
    elif default == "yes":
        suffix = " [Y/n] "
    elif default == "no":
        suffix = " [y/N] "

    # Wait for valid user input, if received return `True`/`False`
    while True:
        choice = input(msg + suffix).lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid.keys():
            return valid[choice]
        else:
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


def user_input(msg, valid, default=None):
    """
    Docstring

    msg = str (message)
    valid = list (avail. options)
    default = str (default option)
    """

    # Add trailing whitespace to `msg` if not already present and append
    # default reply (if provided)
    suffix = "" + " " * (not msg.endswith(" "))
    if default is not None:
        default = default.replace("[", "").replace("]","")
        assert default in valid
        suffix = "[Default: '{}'] ".format(default)

    # Wait for valid user input and return choice upon receipt
    while True:
        choice = input(msg + suffix)
        if default is not None and choice == "":
            return default
        elif choice in valid:
            return choice
        else:
            print("Please respond with '" + \
                  "or '".join(opt + "' " for opt in valid) + "\n")



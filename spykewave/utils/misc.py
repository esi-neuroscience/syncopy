# misc.py - Collection of utility classes/functions for SpykeWave
# 
# Author: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Created: Januar 14 2019
# Last modified: <2019-01-23 15:33:59>

# Builtin/3rd party package imports
import sys

__all__ = ["SPWTypeError", "SPWValueError", "SPWIOError", "spw_print", "spw_warning"]

class Error(Exception):
    """
    Base class for SpykeWave errors
    """
    pass

class SPWTypeError(Error):
    """
    SpykeWave-specific version of a TypeError

    Attributes
    ----------
    var : object
        The culprit responsible for ending up here
    varname : str
        Name of the variable in the code
    expected : type or str
        Expected type of `var`
    """
    
    def __init__(self, var, varname="", expected=""):
        self.found = str(type(var).__name__)
        self.varname = str(varname)
        self.expected = str(expected)
        
    def __str__(self):
        msg = "Wrong type{vn:s}{ex:s}{fd:s}"
        return msg.format(vn=" of " + self.varname + ":" if len(self.varname) else ":",
                          ex=" expected " + self.expected if len(self.expected) else "",
                          fd=" found " + self.found)

class SPWValueError(Error):
    """
    SpykeWave-specific version of a ValueError

    Attributes
    ----------
    legal : str
        Valid value(s) of object
    varname : str
        Name of variable in question
    actual : str
        Actual value of object
    """
    
    def __init__(self, legal, varname="", actual=""):
        self.legal = str(legal)
        self.varname = str(varname)
        self.actual = str(actual)
        
    def __str__(self):
        msg = "Invalid value{vn:s}{fd:s} expected {ex:s}"
        return msg.format(vn=" of " + self.varname + ":" if len(self.varname) else ":",
                          fd=" `" + self.actual + "`;" if len(self.actual) else "",
                          ex=self.legal)

class SPWIOError(Error):
    """
    SpykeWave-specific version of an IO/OSError

    Attributes
    ----------
    fs_loc : str
        File-system location (file/directory) that caused the exception
    exists : bool
        If `exists = True`, `fs_loc` already exists and cannot be overwritten, 
        otherwise `fs_loc` does not exist and hence cannot be read. 
    """
    
    def __init__(self, fs_loc, exists=None):
        self.fs_loc = str(fs_loc)
        self.exists = exists
        
    def __str__(self):
        msg = "Cannot {op:s} {fs_loc:s}{ex:s}"
        return msg.format(op="access" if self.exists is None else "write" if self.exists else "read",
                          fs_loc=self.fs_loc,
                          ex=": object already exists" if self.exists is True \
                          else ": object does not exist" if self.exists is False else "")
    
##########################################################################################
def spw_print(message, caller=sys._getframe().f_back.f_code.co_name):
    """
    Custom print function for package-wide standardized CL output

    Parameters
    ----------
    msg : str
        String to be printed
    caller : str
        Name of calling routine or module to pre-pend `msg` with

    Notes
    -----
    This is an internal routine solely intended to be used by package components. 
    Thus, this funciton does not perform any error-checking and assumes 
    the caller knows what it is doing. 
    """

    msg = "{cl:s}{sep:s}{msg:s}"
    print(msg.format(cl=caller, sep=": " if len(caller) > 0 else "", msg=message))
    return 

##########################################################################################
def spw_warning(message, caller=sys._getframe().f_back.f_code.co_name):
    """
    Custom print function for package-wide standardized warning messages

    Parameters
    ----------
    msg : str
        Warning message
    caller : str
        Name of calling routine or module to pre-pend `msg` with

    Notes
    -----
    This is an internal routine solely intended to be used by package components. 
    Thus, this funciton does not perform any error-checking and assumes 
    the caller knows what it is doing. 
    """

    msg = "{sep1:s}{cl:s}{sep2:s}WARNING: {msg:s}"
    print(msg.format(cl=caller,
                     sep1="<" if len(caller) > 0 else "", 
                     sep2="> " if len(caller) > 0 else "",
                     msg=message))
    return

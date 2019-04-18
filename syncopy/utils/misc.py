# -*- coding: utf-8 -*-
#
# Collection of utility classes/functions for SynCoPy
# 
# Created: 2019-01-14 10:23:44
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-04-18 17:10:05>

# Builtin/3rd party package imports
import sys

__all__ = ["SPYTypeError", "SPYValueError", "SPYIOError", "SPYExceptionHandler", "get_caller"]


class Error(Exception):
    """
    Base class for SynCoPy errors
    """
    pass


class SPYTypeError(Error):
    """
    SynCoPy-specific version of a TypeError

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

    
class SPYValueError(Error):
    """
    SynCoPy-specific version of a ValueError

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

    
class SPYIOError(Error):
    """
    SynCoPy-specific version of an IO/OSError

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

    
def SPYExceptionHandler(exc_type, exc_val, exc_trace):
    """
    Docstring coming soon(ish)...
    """

    # Pass KeyboardInterrupt on to regular excepthook so that CTRL + C
    # can still be used to abort program execution
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_val, exc_trace)
        return

    msg = "some text \n" +\
          "{etype:s} \n" +\
          "{eval:s} \n" +\
          "{trace:s}" +\
          "more info..."

    print("HEEEEEEEEEEEEEEEEEEEEEEEre")
    
    # print(msg.format(etype=str(exc_type),
    #                  eval=str(exc_val),
    #                  trace=str(exc_trace)))

    
def get_caller():
    """
    A very elaborate docstring...
    """
    return sys._getframe().f_back.f_code.co_name

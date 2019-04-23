# -*- coding: utf-8 -*-
#
# Collection of utility classes/functions for SynCoPy
# 
# Created: 2019-01-14 10:23:44
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-04-23 16:16:52>

# Builtin/3rd party package imports
import sys
import traceback
from collections import OrderedDict

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

    
def SPYExceptionHandler(*excargs):
    """
    Docstring coming soon(ish)...
    """

    # Max. depth for printing traceback information
    max_count = 5

    # Depending on the number of input arguments, we're either in Jupyter/iPython
    # or "regular" Python - this matters for coloring error messages
    if len(excargs) == 3:
        isipy = False
        etype, evalue, etb = excargs
    else:
        isipy = True
        etype, evalue, etb = sys.exc_info()
        cols = get_ipython().InteractiveTB.Colors
        cols.filename = cols.filenameEm
        # self, etype, evalue, etb = excargs

    # Pass KeyboardInterrupt on to regular excepthook so that CTRL + C
    # can still be used to abort program execution (only relevant in "regular"
    # Python prompts
    if issubclass(etype, KeyboardInterrupt) and not isipy:
        sys.__excepthook__(etype, evalue, etb)
        return

    # Starty by putting together first line of error message
    emsg = "{}\nSyNCoPy encountered an error in{} \n\n".format(cols.topline if isipy else "",
                                                            cols.Normal if isipy else "")

    # Build an ordered(!) dictionary that encodes separators for traceback components
    sep = OrderedDict({"filename": ", line ",
                       "lineno": " in ",
                       "name": "\n\t",
                       "line": "\n"})

    # Go through traceback and put together a list of strings for printing
    tb_list = []
    tb_count = 0
    for frame in traceback.extract_tb(etb):
        if frame.filename.find("site-packages") < 0:
            tb_entry = ""
            for attr in sep.keys():
                tb_entry += "{}{}{}{}".format(getattr(cols, attr) if isipy else "",
                                              getattr(frame, attr),
                                              cols.Normal if isipy else "",
                                              sep.get(attr))
            if tb_count == 0:
                tb_entry += "\n" + "-"*30 + "\n"
            tb_list.append(tb_entry)
            tb_count += 1
            if tb_count == max_count:
                break
    emsg += "".join(tb_list)

    # Format the exception-part of the traceback - the resulting list usually
    # contains only a single string - if we find more (e.g., in case of a
    # SyntaxError), use everything
    exc_fmt = traceback.format_exception_only(etype, evalue)
    if len(exc_fmt) == 1:
        exc_msg = exc_fmt[0]
        idx = exc_msg.rfind(etype.__name__)
        if idx >= 0:
            exc_msg = exc_msg[idx + len(etype.__name__):]
    else:
        exc_msg = "".join(exc_fmt)

    # Highlight (if we're running in iPython) the name of the encountered exception
    emsg += "\n{}{}{}".format(cols.excName if isipy else "",
                              etype.__name__,
                              cols.Normal if isipy else "")

    # Finally, append the actual exception message and some general info
    emsg += exc_msg
    emsg += "\nUse `sys.last_traceback` to inspect the full error traceback"

    # Show generated message and get outta here
    print(emsg)

    
def get_caller():
    """
    A very elaborate docstring...
    """
    return sys._getframe().f_back.f_code.co_name


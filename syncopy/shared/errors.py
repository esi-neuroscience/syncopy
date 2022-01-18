# -*- coding: utf-8 -*-
#
# Collection of utility classes/functions for Syncopy
#

# Builtin/3rd party package imports
import sys
import traceback
from collections import OrderedDict

# Local imports
from syncopy import __tbcount__

# Custom definition of bold ANSI for formatting errors/warnings in iPython/Jupyter
ansiBold = "\033[1m"

__all__ = []


class SPYError(Exception):
    """
    Base class for SynCoPy errors
    """
    pass


class SPYTypeError(SPYError):
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
        return msg.format(vn=" of `" + self.varname + "`:" if len(self.varname) else ":",
                          ex=" expected " + self.expected if len(self.expected) else "",
                          fd=" found " + self.found)


class SPYValueError(SPYError):
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
        return msg.format(vn=" of `" + self.varname + "`:" if len(self.varname) else ":",
                          fd=" '" + self.actual + "';" if len(self.actual) else "",
                          ex=self.legal)


class SPYIOError(SPYError):
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


class SPYParallelError(SPYError):
    """
    Syncopy-specific error intended for concurrent processing routines

    Attributes
    ----------
    msg : str
        Error message to be printed
    client : dask distributed processing client
        The affected dask client object
    """

    def __init__(self, msg, client=None):
        self.client = client
        self.msg = str(msg)

    def __str__(self):
        if self.client is not None:
            preamble = "Error in distributed computing cluster hosted on {}: "
            preamble = preamble.format(self.client.scheduler_info()["address"])
        else:
            preamble = ""
        err = "{preamble:s}{msg:s}"
        return err.format(preamble=preamble, msg=self.msg)


def SPYExceptionHandler(*excargs, **exckwargs):
    """
    Docstring coming soon(ish)...
    """

    # Depending on the number of input arguments, we're either in Jupyter/iPython
    # or "regular" Python - this matters for coloring error messages
    if len(excargs) == 3:
        isipy = False
        etype, evalue, etb = excargs
    else:
        etype, evalue, etb = sys.exc_info()
        try:                            # careful: if iPython is used to launch a script, ``get_ipython`` is not defined
            ipy = get_ipython()
            isipy = True
            cols = ipy.InteractiveTB.Colors
            cols.filename = cols.filenameEm
            cols.bold = ansiBold
            sys.last_traceback = etb    # smartify ``sys``
        except NameError:
            isipy = False

    # Pass ``KeyboardInterrupt`` on to regular excepthook so that CTRL + C
    # can still be used to abort program execution (only relevant in "regular"
    # Python prompts)
    if issubclass(etype, KeyboardInterrupt) and not isipy:
        sys.__excepthook__(etype, evalue, etb)
        return

    # Starty by putting together first line of error message
    emsg = "{}\nSyNCoPy encountered an error in{} \n\n".format(cols.topline if isipy else "",
                                                               cols.Normal if isipy else "")

    # If we're dealing with a `SyntaxError`, show it and getta outta here
    if issubclass(etype, SyntaxError):

        # Just format exception, don't mess around w/ traceback
        exc_fmt = traceback.format_exception_only(etype, evalue)
        for eline in exc_fmt:
            if "File" in eline:
                eline = eline.split("File ")[1]
                fname, lineno = eline.split(", line ")
                emsg += "{}{}{}".format(cols.filename if isipy else "",
                                        fname,
                                        cols.Normal if isipy else "")
                emsg += ", line {}{}{}".format(cols.lineno if isipy else "",
                                               lineno,
                                               cols.Normal if isipy else "")
            elif "SyntaxError" in eline:
                smsg = eline.split("SyntaxError: ")[1]
                emsg += "{}{}SyntaxError{}: {}{}{}".format(cols.excName if isipy else "",
                                                           cols.bold if isipy else "",
                                                           cols.Normal if isipy else "",
                                                           cols.bold if isipy else "",
                                                           smsg,
                                                           cols.Normal if isipy else "")
            else:
                emsg += "{}{}{}".format(cols.line if isipy else "",
                                        eline,
                                        cols.Normal if isipy else "")

        # Show generated message and leave (or kick-off debugging in Jupyer/iPython if %pdb is on)
        print(emsg)
        if isipy:
            if ipy.call_pdb:
                ipy.InteractiveTB.debugger()
        return

    # Build an ordered(!) dictionary that encodes separators for traceback components
    sep = OrderedDict({"filename": ", line ",
                       "lineno": " in ",
                       "name": "\n\t",
                       "line": "\n"})

    # Find "root" of traceback tree (and remove outer-most frames)
    keepgoing = True
    while keepgoing:
        frame = traceback.extract_tb(etb)[0]
        etb = etb.tb_next
        if frame.filename.find("site-packages") < 0 or \
           (frame.filename.find("site-packages") >= 0 and \
            frame.filename.find("syncopy") >= 0):
            tb_entry = ""
            for attr in sep.keys():
                tb_entry += "{}{}{}{}".format(getattr(cols, attr) if isipy else "",
                                              getattr(frame, attr),
                                              cols.Normal if isipy else "",
                                              sep.get(attr))
            emsg += tb_entry
            keepgoing = False

    # Format the exception-part of the traceback - the resulting list usually
    # contains only a single string - if we find more just use everything
    exc_fmt = traceback.format_exception_only(etype, evalue)
    if len(exc_fmt) == 1:
        exc_msg = exc_fmt[0]
        idx = exc_msg.rfind(etype.__name__)
        if idx >= 0:
            exc_msg = exc_msg[idx + len(etype.__name__):]
        exc_name = "{}{}{}{}".format(cols.excName if isipy else "",
                                     cols.bold if isipy else "",
                                     etype.__name__,
                                     cols.Normal if isipy else "")
    else:
        exc_msg = "".join(exc_fmt)
        exc_name = ""

    # Now go through traceback and put together a list of strings for printing
    if __tbcount__ and etb is not None:
        emsg += "\n" + "-"*80 + "\nAbbreviated traceback:\n\n"
        tb_count = 0
        tb_list = []
        for frame in traceback.extract_tb(etb):
            if frame.filename.find("site-packages") < 0 or \
               (frame.filename.find("site-packages") >= 0 and \
                frame.filename.find("syncopy") >= 0):
                tb_entry = ""
                for attr in sep.keys():
                    tb_entry += "{}{}{}{}".format("", # placeholder for color if wanted
                                                  getattr(frame, attr),
                                                  "", # placeholder for color if wanted
                                                  sep.get(attr))
                tb_list.append(tb_entry)
                tb_count += 1
                if tb_count == __tbcount__:
                    break
        emsg += "".join(tb_list)

    # Finally, another info message
    if etb is not None:
        emsg += "\nUse `import traceback; import sys; traceback.print_tb(sys.last_traceback)` " + \
                "for full error traceback.\n"

    # Glue actual Exception name + message to output string
    emsg += "{}{}{}{}{}".format("\n" if isipy else "",
                                exc_name,
                                cols.bold if isipy else "",
                                exc_msg,
                                cols.Normal if isipy else "",)


    # Show generated message and get outta here
    print(emsg)

    # Kick-start debugging in case %pdb is enabled in Jupyter/iPython
    if isipy:
        if ipy.call_pdb:
            ipy.InteractiveTB.debugger()


def SPYWarning(msg, caller=None):
    """
    Standardized Syncopy warning message

    Parameters
    ----------
    msg : str
        Warning message to be printed
    caller : None or str
        Issuer of warning message. If `None`, name of calling method is
        automatically fetched and pre-pended to `msg`.

    Returns
    -------
    Nothing : None
    """

    # If Syncopy's running in Jupyter/iPython colorize warning message
    # Use the following chart (enter FG color twice b/w ';') to change:
    # https://en.wikipedia.org/wiki/ANSI_escape_code#Colors
    try:
        cols = get_ipython().InteractiveTB.Colors
        warnCol = "\x1b[95;95m"
        normCol = cols.Normal
        boldEm = ansiBold
    except NameError:
        warnCol = ""
        normCol = ""
        boldEm = ""

    # Plug together message string and print it
    if caller is None:
        caller = sys._getframe().f_back.f_code.co_name
    PrintMsg = "{coloron:s}{bold:s}Syncopy{caller:s} WARNING: {msg:s}{coloroff:s}"
    print(PrintMsg.format(coloron=warnCol,
                          bold=boldEm,
                          caller=" <" + caller + ">" if len(caller) else caller,
                          msg=msg,
                          coloroff=normCol))


def SPYInfo(msg, caller=None):
    """
    Standardized Syncopy info message

    Parameters
    ----------
    msg : str
        Info message to be printed
    caller : None or str
        Issuer of info message. If `None`, name of calling method is
        automatically fetched and pre-pended to `msg`.

    Returns
    -------
    Nothing : None
    """

    # If Syncopy's running in Jupyter/iPython colorize warning message
    # Use the following chart (enter FG color twice b/w ';') to change:
    # https://en.wikipedia.org/wiki/ANSI_escape_code#Colors
    try:
        cols = get_ipython().InteractiveTB.Colors
        infoCol = cols.Normal # infos are fine with just bold text
        normCol = cols.Normal
        boldEm = ansiBold
    except NameError:
        infoCol = ""
        normCol = ""
        boldEm = ""

    # Plug together message string and print it
    if caller is None:
        caller = sys._getframe().f_back.f_code.co_name
    PrintMsg = "{coloron:s}{bold:s}Syncopy{caller:s} INFO: {msg:s}{coloroff:s}"
    print(PrintMsg.format(coloron=infoCol,
                          bold=boldEm,
                          caller=" <" + caller + ">" if len(caller) else caller,
                          msg=msg,
                          coloroff=normCol))


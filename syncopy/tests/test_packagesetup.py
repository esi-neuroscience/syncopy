# -*- coding: utf-8 -*-
# 
# Test if Syncopy's basic import setup/tmp storage initialization works as intended
# 
# Created: 2019-11-08 12:20:26
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-11-08 15:30:04>

import os
import shutil
import tempfile
import importlib
import syncopy
from syncopy import __storage__


# check if folder creation in `__storage__` works as expected
def test_storage_access():
    dirNames = [__storage__, "first", "second", "third", "fourth"]
    folderCascade = os.path.join(*dirNames)
    os.makedirs(folderCascade)
    shutil.rmtree(folderCascade)


# check if `SPYTMPDIR` is respected
def test_spytmpdir():
    tmpDir = os.path.join(tempfile.gettempdir(), "spy_storage")
    os.environ["SPYTMPDIR"] = tmpDir
    importlib.reload(syncopy)
    assert syncopy.__storage__ == tmpDir
    shutil.rmtree(tmpDir)


# check if `cleanup` does what it's supposed to do
def test_cleanup():
    pass

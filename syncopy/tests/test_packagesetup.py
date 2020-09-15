# -*- coding: utf-8 -*-
# 
# Test if Syncopy's basic import setup/tmp storage initialization works as intended
# 
# Created: 2019-11-08 12:20:26
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2020-01-16 15:05:20>

import os
import sys
import shutil
import time
import tempfile
import importlib
import subprocess
from glob import glob
import syncopy


# check if folder creation in `__storage__` works as expected
def test_storage_access():
    dirNames = [syncopy.__storage__, "first", "second", "third", "fourth"]
    folderCascade = os.path.join(*dirNames)
    os.makedirs(folderCascade)
    shutil.rmtree(folderCascade)
    time.sleep(1)


# check if `SPYTMPDIR` is respected
def test_spytmpdir():
    tmpDir = os.path.join(tempfile.gettempdir(), "spy_storage")
    os.environ["SPYTMPDIR"] = tmpDir
    importlib.reload(syncopy)
    assert syncopy.__storage__ == tmpDir
    shutil.rmtree(tmpDir)
    del os.environ["SPYTMPDIR"]
    time.sleep(1)


# check if `cleanup` does what it's supposed to do
def test_cleanup():
    # spawn new Python instance, which creates and saves an `AnalogData` object 
    # in custom $SPYTMPDIR; force-kill the process after a few seconds preventing 
    # Syncopy from cleaning up its temp storage folder
    tmpDir = os.path.join(tempfile.gettempdir(), "spy_zombie")
    os.environ["SPYTMPDIR"] = tmpDir
    commandStr = \
        "import os; " +\
        "import time; " +\
        "import syncopy as spy; " +\
        "from syncopy.tests.misc import generate_artificial_data; " +\
        "dummy = generate_artificial_data(inmemory=False); " +\
        "dummy.save(os.path.join(spy.__storage__, 'spy_dummy')); " +\
        "time.sleep(100)"
    process = subprocess.Popen([sys.executable, "-c", commandStr])
    time.sleep(5.)
    process.kill()
    
    # get inventory of external Syncopy instance's temp storage
    spyGarbage = glob(os.path.join(tmpDir, "*"))
    assert len(spyGarbage)

    # launch 2nd external instance with same $SPYTMPDIR, create 2nd `AnalogData` 
    # object, run `cleanup` and keep instance alive in background (for max. 100s)
    commandStr = \
        "import time; " +\
        "import syncopy as spy; " +\
        "from syncopy.tests.misc import generate_artificial_data; " +\
        "dummy = generate_artificial_data(inmemory=False); " +\
        "spy.cleanup(older_than=0, interactive=False); " +\
        "time.sleep(100)"
    process2 = subprocess.Popen([sys.executable, "-c", commandStr], 
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True)
    time.sleep(5)
    
    # ensure `cleanup` call removed first instance's garbage but 2nd `AnalogData` 
    # belonging to 2nd instance launched above is unharmed
    for garbage in spyGarbage:
        assert not os.path.exists(garbage)
    assert glob(os.path.join(tmpDir, "*.analog"))
    
    # now kill 2nd instance and wipe `tmpDir`
    process2.kill()    
    shutil.rmtree(tmpDir)
    del os.environ["SPYTMPDIR"]
    time.sleep(1)

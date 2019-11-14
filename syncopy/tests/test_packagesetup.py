# -*- coding: utf-8 -*-
# 
# Test if Syncopy's basic import setup/tmp storage initialization works as intended
# 
# Created: 2019-11-08 12:20:26
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-11-11 15:39:31>

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


# check if `SPYTMPDIR` is respected
def test_spytmpdir():
    tmpDir = os.path.join(tempfile.gettempdir(), "spy_storage")
    os.environ["SPYTMPDIR"] = tmpDir
    importlib.reload(syncopy)
    assert syncopy.__storage__ == tmpDir
    shutil.rmtree(tmpDir)


# check if `cleanup` does what it's supposed to do
def test_cleanup():
    # run script that spawns new Python instance, which creates and saves an 
    # `AnalogData` object in custom $SPYTMPDIR; force-kill the process after 
    # 10 seconds preventing Syncopy from cleaning up its temp storage folder
    tmpDir = os.path.join(tempfile.gettempdir(), "spy_zombie")
    os.environ["SPYTMPDIR"] = tmpDir
    process = subprocess.Popen([sys.executable, "_zombie_spawner.py"])
    time.sleep(10)
    process.kill()    
    timeout = 20
    counter = 0
    while process.poll() != 1 and counter < timeout:
        time.sleep(0.5)
        counter += 1
    if counter == timeout:
        raise TimeoutError("Python Process {} could not be killed. ".format(process.pid))
    
    # get inventory of external Syncopy instance's temp storage
    spyGarbage = glob(os.path.join(tmpDir, "*"))
    assert len(spyGarbage)
    
    # launch 2nd external instance with same $SPYTMPDIR, create 2nd `AnalogData` 
    # object and run `cleanup`
    commandStr = \
        "import syncopy as spy; " +\
        "from syncopy.tests.misc import generate_artificial_data; " +\
        "dummy = generate_artificial_data(); " +\
        "spy.cleanup(older_than=0, interactive=False); " +\
        "print(dummy.filename)"
    out, err = subprocess.Popen([sys.executable, "-c", commandStr], 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, shell=True).communicate()
    
    # if `out` is empty, something went wrong in instance no. 2 above
    out = out.split()
    assert out
    dummy_filename = out[-1]
    assert tmpDir in dummy_filename
    
    # Ensure `cleanup` call removed first instance's garbage but 2nd `AnalogData` 
    # created above is unharmed
    for garbage in spyGarbage:
        assert not os.path.exists(garbage)
    assert os.path.isfile(dummy_filename)
    
    shutil.rmtree(tmpDir)

# -*- coding: utf-8 -*-
#
# Test if Syncopy's basic import setup/tmp storage initialization works as intended
#

# Builtin/3rd party package imports
import os
import sys
import shutil
import time
import tempfile
import importlib
import subprocess
import pytest
from glob import glob

# Local imports
import syncopy

# Decorator to decide whether or not to run dask-related tests
skip_in_ghactions = pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ.keys(), reason="Do not execute by GitHub Actions")


# check if folder creation in `__storage__` works as expected
def test_storage_access():
    dirNames = [syncopy.__storage__, "first", "second", "third", "fourth"]
    folderCascade = os.path.join(*dirNames)
    os.makedirs(folderCascade)
    shutil.rmtree(folderCascade)
    time.sleep(1)


# check if `SPYTMPDIR` is respected
def test_spytmpdir():
    tmpDir = os.path.join(syncopy.__storage__, "__testStorage__")
    os.environ["SPYTMPDIR"] = tmpDir
    importlib.reload(syncopy)
    assert syncopy.__storage__ == tmpDir
    shutil.rmtree(tmpDir, ignore_errors=True)
    del os.environ["SPYTMPDIR"]
    time.sleep(1)


# check if `cleanup` does what it's supposed to do
@skip_in_ghactions
def test_cleanup():
    # spawn new Python instance, which creates and saves an `AnalogData` object
    # in custom $SPYTMPDIR; force-kill the process after a few seconds preventing
    # Syncopy from cleaning up its temp storage folder

    # this function assumes to be in the root directory of the repository
    # if that is not the case we have to move there

    cdir = os.getcwd()
    while 'syncopy' in cdir:
        head, tail = os.path.split(cdir)
        if not 'syncopy' in head:
            root_dir = os.path.join(head, tail)
            os.chdir(root_dir)
            break
        cdir = head
    # check that we are not entirely somewhere else
    else:
        assert False

    with tempfile.TemporaryDirectory() as tmpDir:

        os.environ["SPYTMPDIR"] = tmpDir
        commandStr = \
            "import os; " +\
            "import time; " +\
            "import numpy as np; " +\
            "import syncopy as spy; " +\
            "dummy = spy.AnalogData(data=np.ones((10,10)), samplerate=1); " +\
            "dummy.save(os.path.join(spy.__storage__, 'spy_dummy')); " +\
            "time.sleep(100)"
        process = subprocess.Popen([sys.executable, "-c", commandStr])
        time.sleep(8)
        process.kill()

        # get inventory of external Syncopy instance's temp storage
        spyGarbage = glob(os.path.join(tmpDir, "*"))

        assert len(spyGarbage)

        # launch 2nd external instance with same $SPYTMPDIR, create 2nd `AnalogData`
        # object, run `cleanup` and keep instance alive in background (for max. 100s)
        commandStr = \
            "import time; " +\
            "import syncopy as spy; " +\
            "import numpy as np; " +\
            "dummy = spy.AnalogData(data=np.ones((10,10)), samplerate=1); " +\
            "spy.cleanup(older_than=0, interactive=False); " +\
            "time.sleep(100)"
        process2 = subprocess.Popen([sys.executable, "-c", commandStr],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True)
        time.sleep(8)

        # ensure `cleanup` call removed first instance's garbage but 2nd `AnalogData`
        # belonging to 2nd instance launched above is unharmed
        for garbage in spyGarbage:
            assert not os.path.exists(garbage)
        assert glob(os.path.join(tmpDir, "*.analog"))

        # now kill 2nd instance and wipe `tmpDir`
        process2.kill()
        time.sleep(1)

    del os.environ["SPYTMPDIR"]
    time.sleep(1)

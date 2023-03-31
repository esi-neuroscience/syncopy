# -*- coding: utf-8 -*-
#
# Test proper functionality of Syncopy's `ContinuousData` class + subclasses
#

# Builtin/3rd party package imports
import os
import tempfile

# Local imports
from syncopy.datatype.util import get_dir_size


class TestDirSize():

    def test_dirsize(self):
        with tempfile.TemporaryDirectory() as tdir:
            fname = "tmpfile"
            for file_idx in range(20):
                tf = os.path.join(tdir, fname + str(file_idx))
                with open(tf, "w") as f:
                    f.write(f"This is a dummy file {file_idx}.")
            dir_size_byte, num_files = get_dir_size(tdir, out="byte")
            assert num_files == 20
            assert dir_size_byte > 200
            assert dir_size_byte < 2000
            assert dir_size_byte == 470
            dir_size_gb, num_files = get_dir_size(tdir, out="GB")
            assert dir_size_gb < 1e-6




if __name__ == '__main__':

    T1 = TestDirSize()


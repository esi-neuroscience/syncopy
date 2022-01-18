# -*- coding: utf-8 -*-
#
# Test proper functionality of Syncopy's decorator mechanics
#

# Builtin/3rd party package imports
import string
import pytest
from syncopy.shared.kwarg_decorators import unwrap_cfg, unwrap_select
from syncopy.tests.misc import generate_artificial_data
from syncopy.shared.tools import StructDict
from syncopy.shared.errors import SPYValueError, SPYTypeError, SPYError


@unwrap_cfg
@unwrap_select
def group_objects(*data, groupbychan=None, select=None):
    """
    Dummy function that collects the `filename` property of all
    input objects that contain a specific channel given by
    `groupbychan`
    """
    group = []
    if groupbychan:
        for obj in data:
            if groupbychan in obj.channel:
                group.append(obj.filename)
    else:
        group = [obj.filename for obj in data]
    return group


class TestSpyCalls():

    nChan = 13
    nObjs = nChan

    # Generate `nChan` objects whose channel-labeling scheme obeys:
    # ob1.channel =  ["A", "B", "C", ..., "M"]
    # ob2.channel =  [     "B", "C", ..., "M", "N"]
    # ob3.channel =  [          "C", ..., "M", "N", "O"]
    # ...
    # ob13.channel = [                    "M", "N", "O", ..., "Z"]
    # Thus, channel no. 13 ("M") is common across all objects
    dataObjs = []
    for n in range(nObjs):
        obj = generate_artificial_data(nChannels=nChan, inmemory=False)
        obj.channel = list(string.ascii_uppercase[n : nChan + n])
        dataObjs.append(obj)
    data = dataObjs[0]

    def test_validcallstyles(self):

        # data positional
        fname, = group_objects(self.data)
        assert fname == self.data.filename

        # data as keyword
        fname, = group_objects(data=self.data)
        assert fname == self.data.filename

        # data in cfg
        cfg = StructDict()
        cfg.data = self.data
        fname, = group_objects(cfg)
        assert fname == self.data.filename

        # 1. data positional, 2. cfg positional
        cfg = StructDict()
        cfg.groupbychan = None
        fname, = group_objects(self.data, cfg)
        assert fname == self.data.filename

        # 1. cfg positional, 2. data positional
        fname, = group_objects(cfg, self.data)
        assert fname == self.data.filename

        # data positional, cfg as keyword
        fname, = group_objects(self.data, cfg=cfg)
        assert fname == self.data.filename

        # cfg positional, data as keyword
        fname, = group_objects(cfg, data=self.data)
        assert fname == self.data.filename

        # both keywords
        fname, = group_objects(cfg=cfg, data=self.data)
        assert fname == self.data.filename

    def test_invalidcallstyles(self):

        # expected error messages
        errmsg1 = "expected Syncopy data object(s) provided either via " +\
                 "`cfg`/keyword or positional arguments, not both"
        errmsg2 = "expected Syncopy data object(s) provided either via `cfg` " +\
            "or as keyword argument, not both"
        errmsg3 = "expected either 'data' or 'dataset' in `cfg`/keywords, not both"

        # ensure things break reliably for 'data' as well as 'dataset'
        for key in ["data", "dataset"]:

            # data + cfg w/data
            cfg = StructDict()
            cfg[key] = self.data
            with pytest.raises(SPYValueError) as exc:
                group_objects(self.data, cfg)
            assert errmsg1 in str(exc.value)

            # data as positional + kwarg
            with pytest.raises(SPYValueError) as exc:
                group_objects(self.data, data=self.data)
            assert errmsg1 in str(exc.value)
            with pytest.raises(SPYValueError) as exc:
                group_objects(self.data, dataset=self.data)
            assert errmsg1 in str(exc.value)

            # cfg w/data + kwarg + positional
            with pytest.raises(SPYValueError) as exc:
                group_objects(self.data, cfg, data=self.data)
            assert errmsg1 in str(exc.value)
            with pytest.raises(SPYValueError) as exc:
                group_objects(self.data, cfg, dataset=self.data)
            assert errmsg1 in str(exc.value)

            # cfg w/data + kwarg
            with pytest.raises(SPYValueError) as exc:
                group_objects(cfg, data=self.data)
            assert errmsg2 in str(exc.value)
            with pytest.raises(SPYValueError) as exc:
                group_objects(cfg, dataset=self.data)
            assert errmsg2 in str(exc.value)

        # cfg (no data) but double-whammied
        cfg = StructDict()
        cfg.groupbychan = None
        with pytest.raises(SPYValueError)as exc:
            group_objects(self.data, cfg, cfg=cfg)
        assert "expected `cfg` either as positional or keyword argument, not both" in str(exc.value)

        # keyword set via cfg and kwarg
        with pytest.raises(SPYValueError) as exc:
            group_objects(self.data, cfg, groupbychan="invalid")
        assert "'non-default value for groupbychan'; expected no keyword arguments" in str(exc.value)

        # both data and dataset in cfg/keywords
        cfg = StructDict()
        cfg.data = self.data
        cfg.dataset = self.data
        with pytest.raises(SPYValueError)as exc:
            group_objects(cfg)
        assert errmsg3 in str(exc.value)
        with pytest.raises(SPYValueError)as exc:
            group_objects(data=self.data, dataset=self.data)
        assert errmsg3 in str(exc.value)

        # data/dataset do not contain Syncopy object
        with pytest.raises(SPYError)as exc:
            group_objects(data="invalid")
        assert "`data` must be Syncopy data object(s)!" in str(exc.value)

        # cfg is not dict/StructDict
        with pytest.raises(SPYTypeError)as exc:
            group_objects(cfg="invalid")
        assert "Wrong type of `cfg`: expected dictionary-like" in str(exc.value)

    def test_varargin(self):

        # data positional
        allFnames = group_objects(*self.dataObjs)
        assert allFnames == [obj.filename for obj in self.dataObjs]

        # data in cfg
        cfg = StructDict()
        cfg.data = self.dataObjs
        fnameList = group_objects(cfg)
        assert allFnames == fnameList

        # group objects by single-letter "channels" in various ways
        for letter in ["L", "E", "I", "A"]:
            letterIdx = string.ascii_uppercase.index(letter)
            nOccurences = letterIdx + 1

            # data positional + keyword to get "reference"
            groupList = group_objects(*self.dataObjs, groupbychan=letter)
            assert len(groupList) == nOccurences

            # 1. data positional, 2. cfg positional
            cfg = StructDict()
            cfg.groupbychan = letter
            fnameList = group_objects(*self.dataObjs, cfg)
            assert groupList == fnameList

            # 1. cfg positional, 2. data positional
            fnameList = group_objects(cfg, *self.dataObjs)
            assert groupList == fnameList

            # data positional, cfg as keyword
            fnameList = group_objects(*self.dataObjs, cfg=cfg)
            assert groupList == fnameList

            # cfg w/data + keyword
            cfg = StructDict()
            cfg.dataset = self.dataObjs
            cfg.groupbychan = letter
            fnameList = group_objects(cfg)
            assert groupList == fnameList

            # data positional + select keyword
            fnameList = group_objects(*self.dataObjs[:letterIdx + 1],
                                       select={"channels": [letter]})
            assert groupList == fnameList

            # data positional + cfg w/select
            cfg = StructDict()
            cfg.select = {"channels": [letter]}
            fnameList = group_objects(*self.dataObjs[:letterIdx + 1], cfg)
            assert groupList == fnameList

            # cfg w/data + select
            cfg = StructDict()
            cfg.data = self.dataObjs[:letterIdx + 1]
            cfg.select = {"channels": [letter]}
            fnameList = group_objects(cfg)
            assert groupList == fnameList

        # invalid selection
        with pytest.raises(SPYValueError) as exc:
            group_objects(*self.dataObjs, select={"channels": ["Z"]})
        assert "expected list/array of channel existing names or indices" in str(exc.value)

        # data does not only contain Syncopy objects
        cfg = StructDict()
        cfg.data = self.dataObjs + ["invalid"]
        with pytest.raises(SPYError)as exc:
            group_objects(cfg)
        assert "`data` must be Syncopy data object(s)!" in str(exc.value)


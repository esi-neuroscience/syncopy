# -*- coding: utf-8 -*-
# 
# 
# 
# Created: 2020-04-22 09:17:57
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2020-04-23 17:50:10>

import string
import pytest
from syncopy.shared.kwarg_decorators import unwrap_cfg, unwrap_select
from syncopy.tests.misc import generate_artificial_data
from syncopy.shared.tools import StructDict
from syncopy.shared.errors import SPYValueError, SPYTypeError, SPYError


@unwrap_cfg
@unwrap_select
def group_objects(data, groupbychan=None, select=None):
    group = []
    if groupbychan:
        for obj in data:
            if groupbychan in obj.channel:
                group.append(obj.filename)
    else:
        group = [obj.filename for obj in data]
    return group

# test `data` provided via positional args + cfg -> must trigger SPYValueError!
# test cfg w/data + dataset -> error

class TestSpyCalls():
    
    nChan = 13
    nObjs = nChan
    
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
        
        # 1. data positional, 2. cfg
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
        
        # # cfg is not dict/StructDict
        # with pytest.raises(SPYTypeError)as exc: # FIXME
        #     group_objects(cfg="invalid")

        # no data input whatsoever        
        with pytest.raises(SPYError)as exc:
            group_objects("invalid")
        assert "missing mandatory argument: `data`" in str(exc.value)
        
    def test_singleobjcall(self):
        pass
    # group_objects(self.dataObjs[0], data=self.dataObjs[0], groupbychan=letter) MUST NOT WORK
    # group_objects(data=self.dataObjs[0], groupbychan=letter) WORKS
    # data + dataset in call/cfg
        
    def test_varargin(self):
        
        for letter in ["L"]:
        # for letter in ["L", "E", "I", "A"]:
            letterIdx = string.ascii_uppercase.index(letter)
            nOccurences = letterIdx + 1
            
            fnameList = group_objects(*self.dataObjs, groupbychan=letter)
            assert len(fnameList) == nOccurences
            
            fnameList2 = group_objects(*self.dataObjs[:letterIdx + 1], 
                                       select={"channels": [letter]})
            assert fnameList == fnameList2
            
            cfg = StructDict()
            cfg.data = self.dataObjs
            cfg.groupbychan = letter
            fnameList3 = group_objects(cfg)
            import pdb; pdb.set_trace()
            assert fnameList == fnameList3
        
            cfg = StructDict()
            cfg.dataset = self.dataObjs
            cfg.groupbychan = letter
            fnameList4 = group_objects(cfg)
            assert fnameList == fnameList4

            cfg = StructDict()
            cfg.data = self.dataObjs[:letterIdx + 1]
            cfg.select = {"channels": [letter]}
            fnameList5 = group_objects(cfg)
            assert fnameList == fnameList5

            cfg = StructDict()
            cfg.data = self.dataObjs[:letterIdx + 1]
            fnameList6 = group_objects(cfg, select={"channels": [letter]})
            assert fnameList == fnameList6
            
            # fnameList3 = group_objects(cfg, select={"channel": letter})


            # !!!!!!!! data contains not onyl syncopy objs

    
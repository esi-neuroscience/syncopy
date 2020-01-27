# -*- coding: utf-8 -*-
# 
# Auxiliaries used across all of Syncopy
# 
# Created: 2020-01-27 13:37:32
# Last modified by: Joscha Schmiedt [joscha.schmiedt@esi-frankfurt.de]
# Last modification time: <2020-01-27 13:43:44>

__all__ = ["StructDict"]

class StructDict(dict):
    """Child-class of dict for emulating MATLAB structs

    Examples
    --------
    cfg = StructDict()
    cfg.a = [0, 25]

    """
    
    def __init__(self, *args, **kwargs):
        """
        Create a child-class of dict whose attributes are its keys
        (thus ensuring that attributes and items are always in sync)
        """
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        
    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        if self.keys():
            ppStr = "Syncopy StructDict\n\n"
            maxKeyLength = max([len(val) for val in self.keys()])
            printString = "{0:>" + str(maxKeyLength + 5) + "} : {1:}\n"
            for key, value in self.items():
                ppStr += printString.format(key, str(value))
            ppStr += "\nUse `dict(cfg)` for copy-paste-friendly format"
        else:
            ppStr = "{}"
        return ppStr

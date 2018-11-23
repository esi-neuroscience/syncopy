#!/usr/bin/make -f
# Waf Makefile wrapper
WAF_HOME=/Users/pantaray/Documents/ESI/SpykeWave/Codes/sfft

all:
	@/Users/pantaray/Documents/ESI/SpykeWave/Codes/sfft/tools/waf build

all-debug:
	@/Users/pantaray/Documents/ESI/SpykeWave/Codes/sfft/tools/waf -v build

all-progress:
	@/Users/pantaray/Documents/ESI/SpykeWave/Codes/sfft/tools/waf -p build

install:
	/Users/pantaray/Documents/ESI/SpykeWave/Codes/sfft/tools/waf install

uninstall:
	/Users/pantaray/Documents/ESI/SpykeWave/Codes/sfft/tools/waf uninstall

clean:
	@/Users/pantaray/Documents/ESI/SpykeWave/Codes/sfft/tools/waf clean

distclean:
	@/Users/pantaray/Documents/ESI/SpykeWave/Codes/sfft/tools/waf distclean
	@-rm -rf build
	@-rm -f Makefile

check:
	@/Users/pantaray/Documents/ESI/SpykeWave/Codes/sfft/tools/waf check

dist:
	@/Users/pantaray/Documents/ESI/SpykeWave/Codes/sfft/tools/waf dist

.PHONY: clean dist distclean check uninstall install all


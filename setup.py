# Builtin/3rd party package imports
import datetime
import ruamel.yaml
from setuptools import setup
import codecs
import os.path

# Local imports
from conda2pip import conda2pip

# Recommended reference implementation taken from PyPA
def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find Syncopy version string.")

# Get package version w/o actually importing it (do not trigger dependency import)
spyVersion = get_version("syncopy/__init__.py")

# Get necessary and optional package dependencies
required, dev = conda2pip(return_lists=True)

# Update citation file
citationFile = "CITATION.cff"
yaml = ruamel.yaml.YAML()
with open(citationFile) as fl:
    ymlObj = yaml.load(fl)
ymlObj["version"] = spyVersion
ymlObj["date-released"] = datetime.datetime.now().strftime("%Y-%m-%d")
with open(citationFile, "w") as fl:
    yaml.dump(ymlObj, fl) 

# Run setup (note: identical arguments supplied in setup.cfg will take precedence)
setup(
    install_requires=required,
    extras_require={"dev": dev},
)

# Builtin/3rd party package imports
import datetime
import ruamel.yaml
from setuptools import setup

# Local imports
from conda2pip import conda2pip
import syncopy

# Get necessary and optional package dependencies
required, dev = conda2pip(return_lists=True)

# Update citation file
citationFile = "CITATION.cff"
yaml = ruamel.yaml.YAML()
with open(citationFile) as fl:
    ymlObj = yaml.load(fl)
ymlObj["version"] = syncopy.__version__
ymlObj["date-released"] = datetime.datetime.now().strftime("%Y-%m-%d")
with open(citationFile, "w") as fl:
    yaml.dump(ymlObj, fl) 

# Run setup (note: identical arguments supplied in setup.cfg will take precedence)
setup(
    install_requires=required,
    extras_require={"dev": dev},
)

# Builtin/3rd party package imports
import datetime
import ruamel.yaml
from setuptools import setup
from setuptools_scm import get_version

# Local imports
from conda2pip import conda2pip

# Get necessary and optional package dependencies
required, dev = conda2pip(return_lists=True)

# Get package version for citationFile (for dev-builds this might differ from
# test-PyPI versions, which are ordered by recency)
spyVersion = get_version(root='.', relative_to=__file__)

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
    # install_requires=required,
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    extras_require={"dev": dev},
)

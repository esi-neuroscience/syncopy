# Builtins
import datetime
from setuptools import setup

# External packages
import ruamel.yaml
from setuptools_scm import get_version

# Local imports
import sys
sys.path.insert(0, ".")
from conda2pip import conda2pip

# Set release version by hand
releaseVersion = "0.21"

# Get necessary and optional package dependencies
required, dev = conda2pip(return_lists=True)

# Get package version for citationFile (for dev-builds this might differ from
# test-PyPI versions, which are ordered by recency)
version = get_version(root='.', relative_to=__file__, local_scheme="no-local-version")

# Release versions (commits at tag) have suffix "dev0": use `releaseVersion` as
# fixed version. for TestPyPI uploads, keep the local `tag.devx` scheme
if version.split(".dev")[-1] == "0":
    versionKws = {"use_scm_version" : False, "version" : releaseVersion}
else:
    versionKws = {"use_scm_version" : {"local_scheme": "no-local-version"}}

# Update citation file
citationFile = "CITATION.cff"
yaml = ruamel.yaml.YAML()
with open(citationFile) as fl:
    ymlObj = yaml.load(fl)
ymlObj["version"] = version
ymlObj["date-released"] = datetime.datetime.now().strftime("%Y-%m-%d")
with open(citationFile, "w") as fl:
    yaml.dump(ymlObj, fl)

# Run setup (note: identical arguments supplied in setup.cfg will take precedence)
setup(
    setup_requires=['setuptools_scm'],
    install_requires=required,
    extras_require={"dev": dev},
    **versionKws
)

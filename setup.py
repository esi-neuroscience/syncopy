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

# Get necessary and optional package dependencies
required, dev = conda2pip(return_lists=True)

# Get package version for citationFile (for dev-builds this might differ from
# test-PyPI versions, which are ordered by recency)
spyVersion = get_version(root='.', relative_to=__file__, local_scheme="no-local-version")

# For release-versions, remove local versioning suffix "dev0"; for TestPyPI uploads,
# keep the local `tag.devx` scheme
versionParts = spyVersion.split(".dev")
if versionParts[-1] == "0":
    spyVersion = "v0.1b1"
    versionKws = {"use_scm_version" : False, "version" : spyVersion}
else:
    versionKws = {"use_scm_version" : {"local_scheme": "no-local-version"}}

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
    setup_requires=['setuptools_scm'],
    install_requires=required,
    extras_require={"dev": dev},
    **versionKws
)

# Builtins
import datetime
from setuptools import setup
import subprocess

# External packages
import ruamel.yaml
from setuptools_scm import get_version

# Local imports
import sys
sys.path.insert(0, ".")
from conda2pip import conda2pip

# Set release version by hand for master branch
releaseVersion = "2022.05"

# Get necessary and optional package dependencies
required, dev = conda2pip(return_lists=True)

# If code has not been obtained via `git` or we're inside the master branch,
# use the hard-coded `releaseVersion` as version. Otherwise keep the local `tag.devx`
# scheme for TestPyPI uploads
proc = subprocess.run("git branch --show-current", shell=True, capture_output=True, text=True)
if proc.returncode !=0 or proc.stdout.strip() == "master":
    version = releaseVersion
    versionKws = {"use_scm_version" : False, "version" : version}
else:
    version = get_version(root='.', relative_to=__file__, local_scheme="no-local-version")
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

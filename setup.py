# Builtin/3rd party package imports
from setuptools import setup

# Local imports
from conda2pip import conda2pip

# Get necessary and optional package dependencies
required, dev = conda2pip(return_lists=True)

# Run setup (note: identical arguments supplied in setup.cfg will take precedence)
setup(
    install_requires=required,
    extras_require={"dev": dev},
)

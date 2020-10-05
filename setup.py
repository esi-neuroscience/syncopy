from setuptools import setup
from conda2pip import conda2pip

# Get necessary and optional package dependencies
required, dev = conda2pip(return_lists=True)

# Run setup (note: identical arguments supplied in setup.cfg will take precedence)
setup(
    setup_requires=['pbr'],
    install_requires=required,
    extras_require={"dev": dev},
    pbr=True
)

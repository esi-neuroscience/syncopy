from setuptools import setup, find_packages

setup(
    name='syncopy',
    version='2023.05',
    packages=find_packages(),
    install_requires=[ "h5py>=2.9", "dask>2022.6", "dask-jobqueue>=0.8", "numpy >=1.10", "scipy>=1.5", "matplotlib>=3.5", "tqdm>=4.31", "natsort>=8.1.0", "psutil>=5.9", "fooof>=1.0" ],
)
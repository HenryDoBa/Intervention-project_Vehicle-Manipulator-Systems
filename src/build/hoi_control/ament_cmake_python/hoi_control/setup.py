from setuptools import find_packages
from setuptools import setup

setup(
    name='hoi_control',
    version='0.1.0',
    packages=find_packages(
        include=('hoi_control', 'hoi_control.*')),
)

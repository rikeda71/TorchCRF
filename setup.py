# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


def _requires_from_file(filename):
    return open(filename).read().splitlines()

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='TorchCRF',
    version='0.0.1',
    description='Implementation of Conditional Random Fields in pytorch',
    long_description=readme,
    author='Ryuya Ikeda',
    author_email='rikeda71@gmail.com',
    install_requires=_requires_from_file('requirements.txt'),
    url='https://github.com/s14t284/TorchCRF',
    license=license,
    packages=find_packages(exclude=('tests', )),
    python_requires='>=3.5',
    test_suite='tests',
)

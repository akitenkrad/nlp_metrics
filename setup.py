# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='nlp_metrics',
    version='0.1.0',
    description='metrics used in Natural Language Processing',
    long_description=readme,
    author='akitenkrad',
    author_email='akitenkrad@gmail.com',
    install_requires=['numpy'],
    dependency_links=[],
    url='',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    test_suite='tests',
)


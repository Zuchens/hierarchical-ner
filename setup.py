import os
import pathlib

import pkg_resources
from setuptools import setup, find_namespace_packages

from setup_utils.mypy import Mypy
from setup_utils.pylint import Pylint
from setup_utils.pytest import PyTest
from setup_utils.yapf import Yapf

with pathlib.Path('requirements.txt').open(encoding='utf-8') as requirements_file:
    install_requires = [str(requirement) for requirement in pkg_resources.parse_requirements(requirements_file)]
    setup(
        name='hierarchical-ner',
        version='2023.0.1',
        description='Rewrite of Hierarchical NER',
        install_requires=install_requires,
        packages=find_namespace_packages(),
        cmdclass={
            'pylint': Pylint,
            'pytest': PyTest,
            'yapf': Yapf,
            'mypy': Mypy,

        },
        scripts=[os.path.join("ner", "src", "train_ner.py")]
    )

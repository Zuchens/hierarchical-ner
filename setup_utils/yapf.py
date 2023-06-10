import os

from setuptools import Command


class Yapf(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system('yapf reformat ner --recursive --in-place')

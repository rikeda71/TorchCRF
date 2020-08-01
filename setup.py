import os
import sys

from setuptools import setup
from setuptools.command.install import install

VERSION = "1.1.0"


def _requires_from_file(filename):
    return open(filename).read().splitlines()


class VerifyReleaseVersion(install):
    def run(self):
        tag = os.getenv("CIRCLE_TAG")

        if tag != VERSION:
            sys.exit(
                "Git tag: {} does not match the version this library: {}".format(
                    tag, VERSION
                )
            )


with open("README.md", "r", encoding="utf-8") as f:
    readme = f.read()


setup(
    name="TorchCRF",
    version=VERSION,
    description="An Implementation of Conditional Random Fields in pytorch",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Ryuya Ikeda",
    author_email="rikeda71@gmail.com",
    install_requires=_requires_from_file("requirements.txt"),
    url="https://github.com/s14t284/TorchCRF",
    license="MIT",
    keywords=["crf", "conditional random fields", "nlp", "natural language processing"],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Text Processing",
    ],
    packages=["TorchCRF"],
    test_suite="tests",
    cmdclass={"verify": VerifyReleaseVersion},
)

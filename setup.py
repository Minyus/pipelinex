import re
from codecs import open
from os import path

from setuptools import find_packages, setup

name = "pipelinex"
here = path.abspath(path.dirname(__file__))

with open(path.join(here, "src", name, "__init__.py"), encoding="utf-8") as f:
    result = re.search(r'__version__ = ["\']([^"\']+)', f.read())

    if not result:
        raise ValueError("Can't find the version")

    version = result.group(1)

setup(
    name=name,
    version=version,
    description="Tool to build production-ready pipelines for experimentation with Kedro and MLflow",
    license="Apache Software License (Apache 2.0)",
    long_description="Please see: https://github.com/Minyus/pipelinex",
    long_description_content_type="text/markdown",
    url="https://github.com/Minyus/pipelinex",
    packages=find_packages(where="src", exclude=["tests"]),
    package_dir={"": "src"},
    test_suite="tests",
    install_requires=["numpy"],
    author="Yusuke Minami",
    author_email="me@minyus.github.com",
    zip_safe=False,
    keywords="pipelines, machine learning, deep learning, data science, data engineering",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
    ],
)

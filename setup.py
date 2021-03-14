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

# get the dependencies and installs
with open("requirements.txt", "r") as f:
    requires = [x.strip() for x in f if x.strip()]

with open("requirements_optional.txt", "r") as f:
    requires_optional = [x.strip() for x in f if x.strip()]
requires_optional += requires

with open("requirements_docs.txt", "r") as f:
    requires_docs = [x.strip() for x in f if x.strip()]
requires_docs += requires_optional

with open("requirements_dev.txt", "r") as f:
    requires_dev = [x.strip() for x in f if x.strip()]
requires_dev += requires_docs

readme = r"""
# PipelineX

GitHub Repository:
https://github.com/Minyus/pipelinex

Documentation:
https://pipelinex.readthedocs.io/
"""

setup(
    name=name,
    version=version,
    description="PipelineX: Python package to build ML pipelines for experimentation with Kedro, MLflow, and more",
    license="Apache Software License (Apache 2.0)",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/Minyus/pipelinex",
    packages=find_packages(where="src", exclude=["tests"]),
    package_dir={"": "src"},
    test_suite="tests",
    install_requires=requires,
    extras_require=dict(
        optional=requires_optional, docs=requires_docs, dev=requires_dev
    ),
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
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ],
)

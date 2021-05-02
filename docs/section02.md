## Install PipelineX

### [Option 1] Install from the PyPI

```bash
pip install pipelinex
```

### [Option 2] Development install 

This is recommended only if you want to modify the source code of PipelineX.

```bash
git clone https://github.com/Minyus/pipelinex.git
cd pipelinex
python setup.py develop
```

### Prepare development environment for PipelineX

You can install packages and organize development environment with [pipenv](https://github.com/pypa/pipenv).
Refer the [pipenv](https://github.com/pypa/pipenv) document to install pipenv.
Once you installed pipenv, you can use pipenv to install and organize your environment.

```sh
# install dependent libraries
$ pipenv install

# install development libraries
$ pipenv install --dev

# install pipelinex
$ pipenv run install

# install pipelinex via setup.py
$ pipenv run install_dev

# lint python code
$ pipenv run lint

# format python code
$ pipenv run fmt

# sort imports
$ pipenv run sort

# apply mypy to python code
$ pipenv run vet

# get into shell
$ pipenv shell

# run test
$ pipenv run test
```

### Prepare Docker environment for PipelineX

```bash
git clone https://github.com/Minyus/pipelinex.git
cd pipelinex
docker build --tag pipelinex .
docker run --rm -it pipelinex
```


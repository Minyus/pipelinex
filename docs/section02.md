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
pip install -e .
```

### Prepare development environment for PipelineX

Use the project test matrix via [tox](https://tox.wiki/) and GitHub Actions to match the optional dependency smoke checks.

Install tox 4.26.0 locally first:

```sh
python -m pip install --upgrade "tox==4.26.0"
```

To run the full listed matrix locally, install the matching Python interpreters for the envs in `tox.ini` first; otherwise only the environments whose interpreters are available on your machine will run.

```sh
# run the optional dependency smoke checks locally
$ python -m tox -e py311-latest
$ python -m tox -e py310-latest
```

See [optional-dependency-matrix.yml](https://github.com/Minyus/pipelinex/blob/master/.github/workflows/optional-dependency-matrix.yml) for the CI matrix.

### Prepare Docker environment for PipelineX

```bash
git clone https://github.com/Minyus/pipelinex.git
cd pipelinex
docker build --tag pipelinex .
docker run --rm -it pipelinex
```

## Install PipelineX

- [Option 1] pip install from the PyPI:

```bash
pip install pipelinex
```

- [Option 2] development install (will be updated as you modify the source code):

```bash
git clone https://github.com/Minyus/pipelinex.git
cd pipelinex
python setup.py develop
```

### Prepare Docker environment for PipelineX

```bash
git clone https://github.com/Minyus/pipelinex.git
cd pipelinex
docker build --tag pipelinex .
docker run --rm -it pipelinex
```


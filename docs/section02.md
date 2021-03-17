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

### Prepare Docker environment for PipelineX

```bash
git clone https://github.com/Minyus/pipelinex.git
cd pipelinex
docker build --tag pipelinex .
docker run --rm -it pipelinex
```


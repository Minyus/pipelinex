# PipelineX Documentation

Available at:
https://pipelinex.readthedocs.io/

## How to generate the Sphinx documentation for Read the Docs

1. Install dependencies
```bash
pip install pipelinex[docs]
```

2. Generate rst (reStructuredText) files for API based on docstrings
```bash
cd <repository>
sphinx-apidoc --module-first -o docs/source/00_api_docs src/pipelinex
```

3. Split the repository top README.md into smaller markdown files

```bash
cd <repository>/docs/source/contents
csplit -f '' -b %02d.md -n 2 ../../../README.md "/^## /" {*}
``` 

4. Add the paths to the markdown files to `index.rst`

```
for i in {0..12}; do printf "   source/contents/%02d.md\n" ${i}; done
```

5. Optional: Generate HTML files to review locally before pushing to the repository

```bash
cd <repository>/docs
make html
```

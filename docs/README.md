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
cd <repository>/docs
sphinx-apidoc -f --module-first -o ./ ../src/pipelinex
```

3. Split the repository top README.md into smaller markdown files

If you use macOS, enable GNU version of csplit:
```bash
brew install coreutils
PATH="$(brew --prefix)/opt/coreutils/libexec/gnubin:$PATH"
```

```bash
cd <repository>/docs
csplit -f 'section' -b %02d.md -n 2 ../README.md "/^## /" {*}
``` 

4. Add the paths to the markdown files to `docs/index.rst`

```
for i in {0..12}; do printf "   section%02d.md\n" ${i}; done
```

5. Optional: Generate HTML files to review locally before pushing to the repository

```bash
cd <repository>/docs
make html
```

# PipelineX Documentation

Available at:
https://pipelinex.readthedocs.io/

## How to generate

1. Install dependencies
```bash
pip install pipelinex[docs]
```

2. Generate rst (reStructuredText) files for API based on docstrings
```bash
cd <repository>
sphinx-apidoc --module-first -o docs/source/00_api_docs src/pipelinex
```

3. Split the repository top README.md

```bash
cd <repository>/docs/source/contents
csplit -f '' -b %02d.md -n 2 ../../../README.md "/^## /" {*}

template=$(cat \
<<HEREDOC
{{title}}
============

.. toctree::
   :maxdepth: 4

   {{title}} <../contents/{{file}}>
HEREDOC
)

for f in $(ls); do t=$(head ${f} -n 1) && t=${t#*\# } && t=$(echo ${t} | tr -d '`' | tr '/' ' ') && r="../toc/${f}.rst" && echo "${template}" > "${r}" && sed -i "s@{{title}}@${t}@g" "${r}" && sed -i "s@{{file}}@${f}@g" "${r}" ; done
``` 

4. Optional: Generate HTML files to review locally before pushing to the repository

```bash
cd <repository>/docs
make html
```

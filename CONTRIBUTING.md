# Your contribution is welcome!

## [Issues](https://github.com/Minyus/pipelinex/issues)

- Question
- Sharing your use case
- Feature Request
- Bug Report

## [Pull requests](https://github.com/Minyus/pipelinex/pulls)

- Improve/refactor the current code (description about how you tested in the pull request comment is appreciated.)
- Improve the documentation
- Add a new feature (e.g. TensorFlow support)
- Add test code (using `pytest`) for the existing/new code to `src/tests` directory

## Testing environment (appreciated)

- Install tox 4.26.0 locally with `python -m pip install --upgrade "tox==4.26.0"`
- Use `python -m tox` to run the optional dependency smoke suite locally, for example `python -m tox -e py310-min-optional`
- Keep local smoke-test runs aligned with the checks used in CI
- CI runs the same matrix defined in `.github/workflows/optional-dependency-matrix.yml`
- Re-run the suite before opening a pull request if you touched optional-dependency paths

## Coding Style (appreciated, but optional)

- Formatting by [Black](https://github.com/psf/black)
- [PEP 484 type hints](https://www.python.org/dev/peps/pep-0484/)
- [Google docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)

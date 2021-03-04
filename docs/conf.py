# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

from recommonmark.transform import AutoStructify

from pipelinex import __version__ as release

# -- Project information -----------------------------------------------------

project = "PipelineX"
copyright = "2021, Yusuke Minami"
author = "Yusuke Minami"

version = release


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "recommonmark",
    "sphinx_copybutton",
]

# enable autosummary plugin (table of contents for modules/classes/class
# methods)
autosummary_generate = True

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The master toctree document.
master_doc = "index"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {"collapse_navigation": False, "style_external_links": True}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "pipelinexdoc"

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "pipelinex", "PipelineX Documentation", [author], 1)]


def remove_arrows_in_examples(lines):
    for i, line in enumerate(lines):
        lines[i] = line.replace(">>>", "")


def autodoc_process_docstring(app, what, name, obj, options, lines):
    remove_arrows_in_examples(lines)


def skip(app, what, name, obj, skip, options):
    if name == "__init__":
        return False
    return skip


def setup(app):
    app.connect("autodoc-process-docstring", autodoc_process_docstring)
    app.connect("autodoc-skip-member", skip)
    # enable rendering RST tables in Markdown
    app.add_config_value("recommonmark_config", {"enable_eval_rst": True}, True)
    app.add_transform(AutoStructify)

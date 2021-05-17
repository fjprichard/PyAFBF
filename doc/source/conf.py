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


# -- Project information -----------------------------------------------------

project = 'PyAFBF'
copyright = 'Frédéric Richard, 2021'
author = 'Frederic Richard'

# The full version, including alpha/beta/rc tags
version = '0.0.1'
html_logo = 'Figures/logo.png'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.githubpages',
              'sphinx.ext.autodoc',
              'sphinxcontrib.bibtex',
              'nbsphinx',
              'sphinx.ext.intersphinx',
              'sphinx_gallery.gen_gallery']

autoclass_content = 'both'
# Bibliography.
bibtex_bibfiles = ['./refs.bib']
bibtex_encoding = 'latin'
bibtex_default_style = 'unsrt'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
show_authors = True

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# Example gallery
sphinx_gallery_conf = {
     'examples_dirs': '../../afbf/Examples',   # path to your example scripts
     'gallery_dirs': 'auto_examples',  # path to where to save gallery
}

# def skip(app, what, name, obj, would_skip, options):
#     if name == "__init__":
#         return False
#     return would_skip


# def setup(app):
#     app.connect("autodoc-skip-member", skip)

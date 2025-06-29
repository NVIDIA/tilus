# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import glob
import shutil
import sphinx_gallery.sorting

import tilus.utils

# from sphinx_gallery.sorting import FileNameSortKey

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'tilus'
copyright = '2025, hidet.org'
author = 'hidet.org'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.coverage',
    'sphinx.ext.todo',
    'sphinx.ext.graphviz',
    'sphinx.ext.doctest',
    'sphinx_copybutton',
    'autodocsumm',
    'sphinx_gallery.gen_gallery'
]

autodoc_typehints = "description"

templates_path = ['_templates']
exclude_patterns = []

sphinx_gallery_conf = {
    'examples_dirs': ['../../examples/matmul'],
    'gallery_dirs': ['getting-started/matmul'],
    'filename_pattern': r'.*\.py',
    'download_all_examples': True,
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_theme_options = {
    "repository_url": "https://github.com/hidet-org/tilus",
    "use_repository_button": True,
    "show_navbar_depth": 1,
}
html_static_path = ['_static']
html_permalinks_icon = "<span>¶</span>"

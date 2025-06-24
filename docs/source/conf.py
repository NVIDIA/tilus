# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

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
]

templates_path = ['_templates']
exclude_patterns = []



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

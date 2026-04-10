# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "tilus"
copyright = "2025, NVIDIA"
author = "NVIDIA"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "sphinx.ext.todo",
    "sphinx.ext.graphviz",
    "sphinx.ext.doctest",
    "sphinx_copybutton",
    "autodocsumm",
]

autodoc_typehints = "description"
autoclass_content = "class"
autodoc_class_signature = "separated"
autodoc_member_order = "alphabetical"

templates_path = ["_templates"]
exclude_patterns = []

intersphinx_mapping = {
    # 'python': ('https://docs.python.org/3', None),
    # 'torch': ('https://pytorch.org/docs/stable/', None), # Use 'stable' or a specific version
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/NVIDIA/tilus",
    "use_repository_button": True,
    "show_navbar_depth": 1,
}
html_sidebars = {
    "**": [
        "navbar-logo.html",
        "icon-links.html",
        "search-button-field.html",
        "sbt-sidebar-nav.html",
        "versioning.html",
    ],
}
html_static_path = ["_static"]
html_css_files = [
    "custom.css",
]
html_permalinks_icon = "<span>¶</span>"

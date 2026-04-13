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

# -- Rewrite autosummary stub titles for instruction groups -------------------


def _build_instruction_group_map():
    """Auto-extract the class-name -> attribute-name mapping from InstructionInterface."""
    from tilus.lang.instructions import InstructionInterface
    from tilus.lang.instructions.base import InstructionGroup

    return {
        type(obj).__name__: attr_name
        for attr_name, obj in vars(InstructionInterface).items()
        if isinstance(obj, InstructionGroup)
    }


_INSTRUCTION_GROUP_MAP = _build_instruction_group_map()


def _rewrite_autosummary_title(app, docname, source):
    """Rewrite page titles of auto-generated stubs.

    The ``source-read`` event fires after autosummary generates stubs but
    before Sphinx parses them, so we can rewrite the RST title in-place.

    - ``tilus.Script.abs`` → ``Script.abs``
    - ``tilus.lang.instructions.mbarrier.BarrierInstructionGroup.arrive`` →
      ``Script.mbarrier.arrive``
    """
    import re

    # Instruction group members: BarrierInstructionGroup.X -> Script.mbarrier.X
    for cls_name, group_name in _INSTRUCTION_GROUP_MAP.items():
        if cls_name + "." in docname:
            member = docname.rsplit(".", 1)[-1]
            new_title = f"Script.{group_name}.{member}"
            source[0] = re.sub(
                r"^.*?\n=+\n",
                new_title + "\n" + "=" * len(new_title) + "\n",
                source[0],
                count=1,
            )
            return

    # Script members: tilus.Script.abs -> Script.abs
    prefix = "tilus.Script."
    if prefix in docname:
        short_name = docname.rsplit(".", 1)[-1]

        new_title = f"Script.{short_name}"
        source[0] = re.sub(
            r"^.*?\n=+\n",
            new_title + "\n" + "=" * len(new_title) + "\n",
            source[0],
            count=1,
        )
        return


def _rewrite_instruction_group_signatures(app, doctree, docname):
    """Rewrite instruction group names in rendered signatures.

    - Class signatures: ``class tilus.lang.instructions.mbarrier.BarrierInstructionGroup``
      → ``class Script.mbarrier``
    - Method signatures: ``BarrierInstructionGroup.arrive(...)``
      → ``Script.mbarrier.arrive(...)``
    """
    from docutils import nodes

    for node in doctree.findall(nodes.Element):
        if node.tagname == "desc_addname":
            text = node.astext()
            for cls_name, group_name in _INSTRUCTION_GROUP_MAP.items():
                if cls_name in text:
                    new_text = text.replace(cls_name, f"Script.{group_name}")
                    node.clear()
                    node += nodes.Text(new_text)
                    break
        elif node.tagname == "desc" and node.get("objtype") == "class":
            # Check if this is an instruction group class
            for sig in node.findall(nodes.Element):
                if sig.tagname != "desc_signature":
                    continue
                for name_node in sig.findall(nodes.Element):
                    if name_node.tagname == "desc_name" and name_node.astext() in _INSTRUCTION_GROUP_MAP:
                        # Replace the desc node with a container holding just the docstring
                        for content in node.findall(nodes.Element):
                            if content.tagname == "desc_content":
                                wrapper = nodes.container()
                                wrapper.extend(content.children)
                                node.replace_self(wrapper)
                                break
                        break
                break


def setup(app):
    app.connect("source-read", _rewrite_autosummary_title)
    app.connect("doctree-resolved", _rewrite_instruction_group_signatures)

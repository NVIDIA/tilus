"""Sphinx extension providing :github:`path` role.

Renders as a link to the file on GitHub with a GitHub icon inside the link.
The base URL is configured via ``github_link_base_url`` in conf.py.

Usage in RST::

    :github:`examples/blackwell_matmul/matmul_v0.py`
"""

from docutils import nodes
from sphinx.application import Sphinx


def github_role(name, rawtext, text, lineno, inliner, options=None, content=None):
    config = inliner.document.settings.env.config
    base_url = config.github_link_base_url.rstrip("/")
    url = f"{base_url}/{text}"

    icon = nodes.raw("", '<i class="fab fa-github"></i> ', format="html")
    link_node = nodes.reference("", "", icon, nodes.Text(text), refuri=url, **{})
    link_node["classes"].append("reference")
    link_node["classes"].append("external")

    return [link_node], []


def setup(app: Sphinx):
    app.add_config_value("github_link_base_url", "", "env")
    app.add_role("github", github_role)
    return {"version": "0.1", "parallel_read_safe": True}

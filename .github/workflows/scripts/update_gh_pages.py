#!/usr/bin/env python3
"""Update versions.json and root index.html for gh-pages deployment."""

import argparse
import json
import pathlib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version-dir", required=True, help="Version directory name (e.g., 'latest' or 'v0.2.0')")
    parser.add_argument("--is-tag", action="store_true", help="Whether this is a tag release")
    args = parser.parse_args()

    version = args.version_dir
    is_tag = args.is_tag

    # Update versions.json
    versions_file = pathlib.Path("versions.json")
    versions = json.loads(versions_file.read_text()) if versions_file.exists() else []
    # versions.json format: [{"slug": "stable", "label": "v0.2.0 (Stable)"}, ...]
    slugs = {v["slug"] for v in versions}

    # Add the version entry if new
    if version not in slugs:
        versions.append({"slug": version, "label": version})
        slugs.add(version)

    # For tag releases, add or update the "stable" entry
    if is_tag:
        versions = [v for v in versions if v["slug"] != "stable"]
        versions.append({"slug": "stable", "label": version + " (Stable)"})

    # Sort: latest first, stable second, then tags in reverse order
    def sort_key(v):
        s = v["slug"]
        if s == "latest":
            return (0, "")
        if s == "stable":
            return (1, "")
        return (2, s)

    versions.sort(key=sort_key)
    versions_file.write_text(json.dumps(versions, indent=2) + "\n")

    # Generate root index.html — redirect to stable if it exists, otherwise latest
    has_stable = any(v["slug"] == "stable" for v in versions)
    target = "stable" if has_stable else "latest"
    pathlib.Path("index.html").write_text(
        "<!DOCTYPE html>\n"
        "<html>\n"
        f'<head><meta http-equiv="refresh" content="0; url={target}/index.html"></head>\n'
        f'<body><p>Redirecting to <a href="{target}/index.html">{target} documentation</a>...</p></body>\n'
        "</html>\n"
    )


if __name__ == "__main__":
    main()

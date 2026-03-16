# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tests that the JavaScript layout demo engine produces identical results to the Python layout implementation.

If these tests fail after changing the Python layout system, the JS demo
at docs/source/_static/layout-demo/layout-demo.js must be updated to match.
"""

from __future__ import annotations

import json
import subprocess
import textwrap
from itertools import product
from pathlib import Path

import pytest
from tilus.ir.layout.ops.register_ops import (
    column_local,
    column_spatial,
    compose,
    divide,
    local,
    reduce,
    reshape,
    spatial,
)
from tilus.ir.layout.register_layout import RegisterLayout

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
JS_ENGINE = REPO_ROOT / "docs" / "source" / "_static" / "layout-demo" / "layout-demo.js"


def _python_layout_to_dict(layout: RegisterLayout) -> dict:
    """Extract the full mapping of a layout as a JSON-serializable dict."""
    return {
        "shape": list(layout.shape),
        "mode_shape": list(layout.modeShape if hasattr(layout, "modeShape") else layout.mode_shape),
        "spatial_modes": list(layout.spatialModes if hasattr(layout, "spatialModes") else layout.spatial_modes),
        "local_modes": list(layout.localModes if hasattr(layout, "localModes") else layout.local_modes),
    }


def _python_grid(layout: RegisterLayout) -> list[dict]:
    """Compute the full grid mapping for a layout using Python."""
    shape = list(layout.shape)
    # Normalize to 2D like the visualizer
    if len(shape) > 3:
        shape = [s for s in shape if s > 1]
        while len(shape) > 3:
            shape = [shape[0] * shape[1]] + shape[2:]
    while len(shape) < 2:
        shape.insert(0, 1)
    layout = layout.with_shape(shape)

    grid = []
    for indices in product(*(range(s) for s in shape)):
        spatial_ids = layout.get_spatial(list(indices))
        spatial_ids = sorted([int(s) for s in spatial_ids])
        local_id = int(layout.get_local(list(indices)))
        grid.append(
            {
                "indices": list(indices),
                "spatial": spatial_ids,
                "local": local_id,
            }
        )
    return grid


def _js_eval(expr: str) -> dict:
    """Evaluate a layout expression in the JS engine and return its attributes + grid."""
    js_code = textwrap.dedent(f"""\
        // Minimal DOM stub
        global.document = {{ readyState: 'loading', addEventListener: function() {{}} }};

        const fs = require('fs');
        let code = fs.readFileSync({str(JS_ENGINE)!r}, 'utf8');

        // Expose internals
        code = code.replace(
            'if (document.readyState',
            'global._parseExpression = parseExpression; if (document.readyState'
        );
        eval(code);

        const layout = _parseExpression({expr!r});

        // Normalize shape for grid computation (same as renderLayout)
        let shape = [...layout.shape];
        if (shape.length > 3) {{
            shape = shape.filter(s => s > 1);
            while (shape.length > 3) shape = [shape[0] * shape[1], ...shape.slice(2)];
        }}
        while (shape.length < 2) shape.unshift(1);
        const displayLayout = layout.withShape(shape);

        // Compute grid
        const grid = [];
        function cartesian(arrays) {{
            if (arrays.length === 0) return [[]];
            const rest = cartesian(arrays.slice(1));
            const result = [];
            for (let i = 0; i < arrays[0]; i++)
                for (const r of rest) result.push([i, ...r]);
            return result;
        }}
        for (const indices of cartesian(shape)) {{
            const spatial = displayLayout.getSpatial(indices);
            const local_id = displayLayout.getLocal(indices);
            grid.push({{ indices, spatial, local: local_id }});
        }}

        console.log(JSON.stringify({{
            attrs: {{
                shape: layout.shape,
                mode_shape: layout.modeShape,
                spatial_modes: layout.spatialModes,
                local_modes: layout.localModes,
            }},
            grid: grid,
        }}));
    """)
    result = subprocess.run(
        ["node", "-e", js_code],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(f"JS evaluation failed for {expr!r}:\n{result.stderr}")
    return json.loads(result.stdout.strip())


# ============================================================
# Test cases: (expression_string, python_layout)
# ============================================================


def _test_cases() -> list[tuple[str, RegisterLayout]]:
    """Return pairs of (JS expression, equivalent Python layout)."""
    return [
        # Basic primitives
        ("spatial(3, 4)", spatial(3, 4)),
        ("local(3, 4)", local(3, 4)),
        ("spatial(2, 3)", spatial(2, 3)),
        ("local(2, 3)", local(2, 3)),
        ("column_spatial(2, 3)", column_spatial(2, 3)),
        ("column_local(2, 3)", column_local(2, 3)),
        ("spatial(4, 8)", spatial(4, 8)),
        ("local(1, 1)", local(1, 1)),
        # Composition via chaining
        ("local(3, 4).spatial(2, 3)", compose(local(3, 4), spatial(2, 3))),
        ("spatial(2, 3).local(3, 4)", compose(spatial(2, 3), local(3, 4))),
        ("spatial(4, 8).local(4, 4)", compose(spatial(4, 8), local(4, 4))),
        # Composition via compose()
        ("compose(local(2, 2), spatial(4, 4))", compose(local(2, 2), spatial(4, 4))),
        # Multi-level composition
        (
            "local(2, 2).spatial(4, 4).local(2, 2)",
            compose(compose(local(2, 2), spatial(4, 4)), local(2, 2)),
        ),
        # Column variants composed
        (
            "column_local(2, 3).column_spatial(4, 2)",
            compose(column_local(2, 3), column_spatial(4, 2)),
        ),
        # Reduce (creates replicated threads)
        ("reduce(spatial(3, 4), [0])", reduce(spatial(3, 4), [0])),
        ("reduce(spatial(4, 6), [1])", reduce(spatial(4, 6), [1])),
        # Divide
        (
            "divide(spatial(4, 8).local(4, 4), local(4, 4))",
            divide(compose(spatial(4, 8), local(4, 4)), local(4, 4)),
        ),
        # Reshape
        (
            "reshape(spatial(4, 8), [8, 4])",
            reshape(spatial(4, 8), [8, 4]),
        ),
        # * operator
        ("local(3, 4) * spatial(2, 3)", compose(local(3, 4), spatial(2, 3))),
        # MMA-like layout (from docs)
        (
            "local(2, 1).spatial(8, 4).local(1, 2)",
            compose(compose(local(2, 1), spatial(8, 4)), local(1, 2)),
        ),
    ]


@pytest.fixture(scope="module")
def check_node():
    """Skip all tests if Node.js is not available."""
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            pytest.skip("Node.js not available")
    except FileNotFoundError:
        pytest.skip("Node.js not installed")


@pytest.mark.parametrize(
    "expr,py_layout",
    _test_cases(),
    ids=[tc[0] for tc in _test_cases()],
)
def test_js_matches_python_attrs(check_node, expr: str, py_layout: RegisterLayout):
    """Test that JS and Python produce the same layout attributes."""
    js_result = _js_eval(expr)
    js_attrs = js_result["attrs"]

    assert js_attrs["shape"] == list(py_layout.shape), (
        f"shape mismatch for {expr}: JS={js_attrs['shape']} Python={list(py_layout.shape)}"
    )
    assert js_attrs["mode_shape"] == list(py_layout.mode_shape), (
        f"mode_shape mismatch for {expr}: JS={js_attrs['mode_shape']} Python={list(py_layout.mode_shape)}"
    )
    assert js_attrs["spatial_modes"] == list(py_layout.spatial_modes), (
        f"spatial_modes mismatch for {expr}: JS={js_attrs['spatial_modes']} Python={list(py_layout.spatial_modes)}"
    )
    assert js_attrs["local_modes"] == list(py_layout.local_modes), (
        f"local_modes mismatch for {expr}: JS={js_attrs['local_modes']} Python={list(py_layout.local_modes)}"
    )


@pytest.mark.parametrize(
    "expr,py_layout",
    _test_cases(),
    ids=[tc[0] for tc in _test_cases()],
)
def test_js_matches_python_grid(check_node, expr: str, py_layout: RegisterLayout):
    """Test that JS and Python produce the same thread/local mapping for every cell."""
    js_result = _js_eval(expr)
    js_grid = js_result["grid"]
    py_grid = _python_grid(py_layout)

    assert len(js_grid) == len(py_grid), f"grid size mismatch for {expr}: JS={len(js_grid)} Python={len(py_grid)}"

    for js_cell, py_cell in zip(js_grid, py_grid):
        assert js_cell["indices"] == py_cell["indices"], f"indices mismatch for {expr}"
        assert js_cell["spatial"] == py_cell["spatial"], (
            f"spatial mismatch at {js_cell['indices']} for {expr}: JS={js_cell['spatial']} Python={py_cell['spatial']}"
        )
        assert js_cell["local"] == py_cell["local"], (
            f"local mismatch at {js_cell['indices']} for {expr}: JS={js_cell['local']} Python={py_cell['local']}"
        )

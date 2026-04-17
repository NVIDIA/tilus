// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Layout math engine + interactive demo for Tilus RegisterLayout
// Port of python/tilus/ir/layout/register_layout.py and ops/register_ops.py

(function () {
  "use strict";

  // ============================================================
  // Utility helpers
  // ============================================================

  function prod(arr) {
    let p = 1;
    for (const v of arr) p *= v;
    return p;
  }

  function gcd(a, b) {
    while (b) {
      [a, b] = [b, a % b];
    }
    return a;
  }

  /** Decompose a flat index into multi-dimensional indices (row-major). */
  function indexDeserialize(index, shape) {
    const indices = new Array(shape.length);
    for (let i = shape.length - 1; i >= 0; i--) {
      indices[i] = index % shape[i];
      index = Math.floor(index / shape[i]);
    }
    return indices;
  }

  /** Serialize multi-dimensional indices to a flat index (row-major). */
  function indexSerialize(indices, shape) {
    let idx = 0;
    for (let i = 0; i < shape.length; i++) {
      idx = idx * shape[i] + indices[i];
    }
    return idx;
  }

  // ============================================================
  // get_mode_groups  (from ops/utils.py)
  // ============================================================

  function getModeGroups(shape, modeShape) {
    let i = 0;
    const groupedModes = [];
    for (const s of shape) {
      const group = [];
      let remaining = s;
      while (remaining > 1) {
        if (i >= modeShape.length || remaining % modeShape[i] !== 0) {
          throw new Error(
            `Cannot group modes for shape [${shape}] with mode_shape [${modeShape}]`
          );
        }
        remaining /= modeShape[i];
        group.push(i);
        i++;
      }
      groupedModes.push(group);
    }
    if (i !== modeShape.length) {
      throw new Error(
        `Cannot group modes for shape [${shape}] with mode_shape [${modeShape}]`
      );
    }
    return groupedModes;
  }

  // ============================================================
  // RegisterLayout
  // ============================================================

  class RegisterLayout {
    constructor(shape, modeShape, spatialModes, localModes) {
      this.shape = shape;
      this.modeShape = modeShape;
      this.spatialModes = spatialModes;
      this.localModes = localModes;
    }

    get groupedModes() {
      if (!this._groupedModes) {
        this._groupedModes = getModeGroups(this.shape, this.modeShape);
      }
      return this._groupedModes;
    }

    get spatialShape() {
      return this.spatialModes.map((m) => (m >= 0 ? this.modeShape[m] : -m));
    }

    get localShape() {
      return this.localModes.map((m) => this.modeShape[m]);
    }

    get spatialSize() {
      return prod(this.spatialShape);
    }

    get localSize() {
      return prod(this.localShape);
    }

    get size() {
      return prod(this.shape);
    }

    withShape(shape) {
      return new RegisterLayout(
        shape,
        this.modeShape,
        this.spatialModes,
        this.localModes
      );
    }

    /** Return sorted list of spatial (thread) indices for a given global index. */
    getSpatial(globalIndices) {
      const modeIndices = [];
      for (let d = 0; d < globalIndices.length; d++) {
        const modes = this.groupedModes[d];
        const ms = modes.map((m) => this.modeShape[m]);
        modeIndices.push(...indexDeserialize(globalIndices[d], ms));
      }

      const replicateDims = [];
      const replicateSizes = [];
      const spatialIndices = [];
      for (let i = 0; i < this.spatialModes.length; i++) {
        const mode = this.spatialModes[i];
        if (mode < 0) {
          replicateDims.push(i);
          replicateSizes.push(-mode);
          spatialIndices.push(0);
        } else {
          spatialIndices.push(modeIndices[mode]);
        }
      }

      const results = [];
      // Cartesian product over replicated dims
      const total = prod(replicateSizes);
      for (let r = 0; r < total; r++) {
        let rem = r;
        for (let k = replicateSizes.length - 1; k >= 0; k--) {
          spatialIndices[replicateDims[k]] = rem % replicateSizes[k];
          rem = Math.floor(rem / replicateSizes[k]);
        }
        results.push(indexSerialize(spatialIndices, this.spatialShape));
      }
      results.sort((a, b) => a - b);
      return results;
    }

    /** Return the local index for a given global index. */
    getLocal(globalIndices) {
      const modeIndices = [];
      for (let d = 0; d < globalIndices.length; d++) {
        const modes = this.groupedModes[d];
        const ms = modes.map((m) => this.modeShape[m]);
        modeIndices.push(...indexDeserialize(globalIndices[d], ms));
      }
      const localIndices = this.localModes.map((m) => modeIndices[m]);
      return indexSerialize(localIndices, this.localShape);
    }

    /** Return the intermediate mapping steps for a given global index. */
    getMappingSteps(globalIndices) {
      // Step 1: global index -> mode indices
      const modeIndices = [];
      for (let d = 0; d < globalIndices.length; d++) {
        const modes = this.groupedModes[d];
        const ms = modes.map((m) => this.modeShape[m]);
        modeIndices.push(...indexDeserialize(globalIndices[d], ms));
      }

      // Step 2: mode indices -> spatial indices (multi-dim) & local indices (multi-dim)
      const spatialIndices = [];
      const spatialLabels = [];
      for (let i = 0; i < this.spatialModes.length; i++) {
        const mode = this.spatialModes[i];
        if (mode < 0) {
          spatialIndices.push("*");
          spatialLabels.push(`rep(${-mode})`);
        } else {
          spatialIndices.push(modeIndices[mode]);
          spatialLabels.push(`m${mode}`);
        }
      }
      const localIndices = this.localModes.map((m) => modeIndices[m]);
      const localLabels = this.localModes.map((m) => `m${m}`);

      // Step 3: flatten
      const spatialFlat = this.getSpatial(globalIndices);
      const localFlat = this.getLocal(globalIndices);

      return {
        globalIndices,
        modeIndices,
        spatialIndices,
        spatialLabels,
        localIndices,
        localLabels,
        spatialFlat,
        localFlat,
      };
    }

    toString() {
      return (
        `RegisterLayout(shape=[${this.shape}], mode_shape=[${this.modeShape}], ` +
        `spatial_modes=[${this.spatialModes}], local_modes=[${this.localModes}])`
      );
    }
  }

  // ============================================================
  // Canonicalize
  // ============================================================

  function canonicalizeSingletonModes(layout) {
    const singletons = layout.modeShape
      .map((s, i) => (s === 1 ? i : -1))
      .filter((i) => i >= 0);
    if (singletons.length === 0) return layout;

    const modeMap = {};
    let idx = 0;
    for (let m = 0; m < layout.modeShape.length; m++) {
      if (layout.modeShape[m] === 1) {
        modeMap[m] = -1;
      } else {
        modeMap[m] = idx++;
      }
    }

    const ms = layout.modeShape.filter((s) => s > 1);
    const sm = layout.spatialModes.filter(
      (m) => m < 0 || modeMap[m] !== -1
    ).map((m) => (m >= 0 ? modeMap[m] : m));
    const lm = layout.localModes
      .filter((m) => modeMap[m] !== -1)
      .map((m) => modeMap[m]);
    return new RegisterLayout(layout.shape, ms, sm, lm);
  }

  function canonicalizeContiguousModes(layout) {
    const modeKind = {};
    const modeIndex = {};
    for (let i = 0; i < layout.spatialModes.length; i++) {
      const m = layout.spatialModes[i];
      if (m < 0) continue;
      modeKind[m] = "spatial";
      modeIndex[m] = i;
    }
    for (let i = 0; i < layout.localModes.length; i++) {
      const m = layout.localModes[i];
      modeKind[m] = "local";
      modeIndex[m] = i;
    }

    const mergeModes = [];
    for (const modes of layout.groupedModes) {
      let i = 0;
      while (i < modes.length) {
        let j = i;
        while (
          j + 1 < modes.length &&
          modeKind[modes[j]] === modeKind[modes[j + 1]] &&
          modeIndex[modes[j]] + 1 === modeIndex[modes[j + 1]]
        ) {
          j++;
        }
        mergeModes.push(modes.slice(i, j + 1));
        i = j + 1;
      }
    }

    if (mergeModes.every((g) => g.length === 1)) return layout;

    const modeMap = {};
    for (let i = 0; i < mergeModes.length; i++) {
      for (let j = 0; j < mergeModes[i].length; j++) {
        modeMap[mergeModes[i][j]] = j === 0 ? i : -1;
      }
    }

    const ms = mergeModes.map((g) =>
      g.reduce((p, m) => p * layout.modeShape[m], 1)
    );
    const sm = layout.spatialModes
      .filter((m) => m < 0 || modeMap[m] !== -1)
      .map((m) => (m >= 0 ? modeMap[m] : m));
    const lm = layout.localModes
      .filter((m) => modeMap[m] !== -1)
      .map((m) => modeMap[m]);
    return new RegisterLayout(layout.shape, ms, sm, lm);
  }

  function canonicalizeLayout(layout) {
    return canonicalizeContiguousModes(canonicalizeSingletonModes(layout));
  }

  // ============================================================
  // Validation + factory
  // ============================================================

  function validateLayout(shape, modeShape, spatialModes, localModes) {
    if (shape.some((s) => s < 1))
      throw new Error("Shape must only be positive integers");

    const remaining = [...shape];
    for (let i = modeShape.length - 1; i >= 0; i--) {
      const mode = modeShape[i];
      if (mode === 1) continue;
      while (remaining.length && remaining[remaining.length - 1] === 1)
        remaining.pop();
      if (!remaining.length || remaining[remaining.length - 1] % mode !== 0) {
        throw new Error(
          `Mode ${mode} does not divide the remaining shape [${remaining}]`
        );
      }
      remaining[remaining.length - 1] /= mode;
    }
    while (remaining.length && remaining[remaining.length - 1] === 1)
      remaining.pop();
    if (remaining.length)
      throw new Error(
        `Modes [${modeShape}] and shape [${shape}] do not match`
      );

    const usedDims = [];
    for (const d of spatialModes) {
      if (d < 0) continue;
      if (d < 0 || d >= modeShape.length)
        throw new Error(`Spatial dim ${d} out of range`);
      usedDims.push(d);
    }
    for (const d of localModes) {
      if (d < 0 || d >= modeShape.length)
        throw new Error(`Local dim ${d} out of range`);
      usedDims.push(d);
    }
    if (new Set(usedDims).size !== usedDims.length)
      throw new Error("Spatial dims and local dims must be unique");
  }

  function registerLayout(shape, modeShape, spatialModes, localModes) {
    validateLayout(shape, modeShape, spatialModes, localModes);
    const layout = new RegisterLayout(shape, modeShape, spatialModes, localModes);
    return canonicalizeLayout(layout);
  }

  // ============================================================
  // Layout operations (from register_ops.py)
  // ============================================================

  function spatial(...args) {
    let shape, ranks;
    if (args.length === 1 && typeof args[0] === "object" && !Array.isArray(args[0])) {
      ({ shape, ranks } = args[0]);
    } else {
      shape = args;
      ranks = null;
    }
    let spatialModes;
    if (ranks) {
      spatialModes = [];
      for (let i = 0; i < shape.length; i++) {
        spatialModes.push(ranks.indexOf(i));
      }
    } else {
      spatialModes = shape.map((_, i) => i);
    }
    return registerLayout(shape, shape, spatialModes, []);
  }

  function local(...args) {
    let shape, ranks;
    if (args.length === 1 && typeof args[0] === "object" && !Array.isArray(args[0])) {
      ({ shape, ranks } = args[0]);
    } else {
      shape = args;
      ranks = null;
    }
    let localModes;
    if (ranks) {
      localModes = [];
      for (let i = 0; i < shape.length; i++) {
        localModes.push(ranks.indexOf(i));
      }
    } else {
      localModes = shape.map((_, i) => i);
    }
    return registerLayout(shape, shape, [], localModes);
  }

  function columnSpatial(...shape) {
    const ranks = [...Array(shape.length).keys()].reverse();
    return spatial({ shape, ranks });
  }

  function columnLocal(...shape) {
    const ranks = [...Array(shape.length).keys()].reverse();
    return local({ shape, ranks });
  }

  function replicated(shape, numWorkers) {
    const replicatedUnit = new RegisterLayout([], [], [-numWorkers], []);
    return compose(replicatedUnit, local(...shape));
  }

  // unsqueeze
  function unsqueeze(layout, dims) {
    if (dims.length === 0) return layout;
    const shape = [];
    let current = 0;
    for (let i = 0; i < layout.shape.length + dims.length; i++) {
      if (dims.includes(i)) {
        shape.push(1);
      } else {
        shape.push(layout.shape[current]);
        current++;
      }
    }
    return layout.withShape(shape);
  }

  // squeeze
  function squeeze(layout, dims) {
    if (dims.length === 0) return layout;
    const shape = layout.shape.filter((_, i) => !dims.includes(i));
    return layout.withShape(shape);
  }

  // compose — also tags the result with operand info for the product explorer
  function compose(outer, inner) {
    const ndims = Math.max(outer.shape.length, inner.shape.length);
    outer = unsqueeze(
      outer,
      [...Array(ndims - outer.shape.length).keys()]
    );
    inner = unsqueeze(
      inner,
      [...Array(ndims - inner.shape.length).keys()]
    );

    const shape = outer.shape.map((a, i) => a * inner.shape[i]);
    const outerMap = {};
    const innerMap = {};
    let currentOuter = 0;
    let currentInner = 0;
    let currentComposed = 0;
    const modeShape = [];

    for (let d = 0; d < ndims; d++) {
      const outerModes = outer.groupedModes[d];
      const innerModes = inner.groupedModes[d];
      for (const om of outerModes) {
        outerMap[currentOuter] = currentComposed;
        currentOuter++;
        currentComposed++;
        modeShape.push(outer.modeShape[om]);
      }
      for (const im of innerModes) {
        innerMap[currentInner] = currentComposed;
        currentInner++;
        currentComposed++;
        modeShape.push(inner.modeShape[im]);
      }
    }

    const spatialModes = [
      ...outer.spatialModes.map((m) => (m >= 0 ? outerMap[m] : m)),
      ...inner.spatialModes.map((m) => (m >= 0 ? innerMap[m] : m)),
    ];
    const localModes = [
      ...outer.localModes.map((m) => outerMap[m]),
      ...inner.localModes.map((m) => innerMap[m]),
    ];

    const result = registerLayout(shape, modeShape, spatialModes, localModes);
    result._productOf = { outer, inner };
    return result;
  }

  // permute
  function permute(layout, dims) {
    const shape = dims.map((d) => layout.shape[d]);
    const groupedModes = dims.map((d) => layout.groupedModes[d]);
    const modeShape = groupedModes.flatMap((g) =>
      g.map((m) => layout.modeShape[m])
    );
    const permutedModes = groupedModes.flatMap((g) => g);
    const modeMap = {};
    permutedModes.forEach((old, newIdx) => {
      modeMap[old] = newIdx;
    });

    return registerLayout(
      shape,
      modeShape,
      layout.spatialModes.map((d) => (d >= 0 ? modeMap[d] : d)),
      layout.localModes.map((d) => modeMap[d])
    );
  }

  // reduce
  function reduce(layout, dims, keepdims = false) {
    if (dims.length === 0) return layout;

    const modesToReduce = [];
    const remainingModes = [];
    for (let i = 0; i < layout.groupedModes.length; i++) {
      for (const m of layout.groupedModes[i]) {
        if (dims.includes(i)) modesToReduce.push(m);
        else remainingModes.push(m);
      }
    }

    const modeMap = {};
    let idx = 0;
    for (let m = 0; m < layout.modeShape.length; m++) {
      if (modesToReduce.includes(m)) {
        modeMap[m] = -1;
      } else {
        modeMap[m] = idx++;
      }
    }

    const spatialModes = [];
    for (const sd of layout.spatialModes) {
      if (sd < 0) {
        spatialModes.push(sd);
      } else {
        if (modeMap[sd] === -1) {
          spatialModes.push(-layout.modeShape[sd]);
        } else {
          spatialModes.push(modeMap[sd]);
        }
      }
    }
    const localModes = layout.localModes
      .filter((m) => modeMap[m] !== -1)
      .map((m) => modeMap[m]);

    let shape;
    if (keepdims) {
      shape = layout.shape.map((s, i) => (dims.includes(i) ? 1 : s));
    } else {
      shape = layout.shape.filter((_, i) => !dims.includes(i));
    }
    const ms = remainingModes.map((m) => layout.modeShape[m]);

    return registerLayout(shape, ms, spatialModes, localModes);
  }

  // reshape
  function reshape(layout, newShape) {
    if (prod(layout.shape) !== prod(newShape))
      throw new Error(
        `Cannot reshape from [${layout.shape}] to [${newShape}]`
      );

    const originalShape = [...newShape];
    layout = canonicalizeLayout(layout);

    const ms = [...layout.modeShape];
    const sh = [...newShape];
    const groupedModeShape = [];

    while (ms.length) {
      let p = ms.shift();
      groupedModeShape.push([]);

      while (sh.length) {
        const q = sh[0];
        if (q % p === 0) {
          groupedModeShape[groupedModeShape.length - 1].push(p);
          sh[0] = q / p;
          if (sh[0] === 1) sh.shift();
          break;
        } else if (p % q === 0) {
          if (q > 1) groupedModeShape[groupedModeShape.length - 1].push(q);
          p = p / q;
          sh.shift();
        } else {
          throw new Error(
            `Cannot reshape layout with shape [${layout.shape}] to [${newShape}]`
          );
        }
      }
    }

    const newModeShape = groupedModeShape.flat();
    const modeMap = {};
    let k = 0;
    for (let mode = 0; mode < groupedModeShape.length; mode++) {
      modeMap[mode] = [];
      for (let j = 0; j < groupedModeShape[mode].length; j++) {
        modeMap[mode].push(k + j);
      }
      k += groupedModeShape[mode].length;
    }

    const spatialModes = [];
    for (const m of layout.spatialModes) {
      if (m < 0) spatialModes.push(m);
      else spatialModes.push(...modeMap[m]);
    }
    const localModes = [];
    for (const m of layout.localModes) {
      localModes.push(...modeMap[m]);
    }

    return registerLayout(originalShape, newModeShape, spatialModes, localModes);
  }

  // flatten
  function flatten(layout, startDim = 0, endDim = -1) {
    if (endDim < 0) endDim = layout.shape.length + endDim;
    const shape = [
      ...layout.shape.slice(0, startDim),
      prod(layout.shape.slice(startDim, endDim)),
      ...layout.shape.slice(endDim),
    ];
    return reshape(layout, shape);
  }

  // divide
  function _layoutWithModeShape(layout, modeShape) {
    const modeMap = {};
    let i = 0;
    for (let mode = 0; mode < layout.modeShape.length; mode++) {
      modeMap[mode] = [];
      let remaining = layout.modeShape[mode];
      while (i < modeShape.length && remaining % modeShape[i] === 0) {
        modeMap[mode].push(i);
        remaining /= modeShape[i];
        i++;
      }
    }
    const spatialModes = [];
    for (const m of layout.spatialModes) {
      if (m < 0) spatialModes.push(m);
      else spatialModes.push(...modeMap[m]);
    }
    const localModes = [];
    for (const m of layout.localModes) {
      localModes.push(...modeMap[m]);
    }
    return new RegisterLayout(layout.shape, modeShape, spatialModes, localModes);
  }

  function divide(lhs, rhs) {
    if (lhs.shape.length < rhs.shape.length)
      throw new Error(`Cannot divide: lhs has fewer dims than rhs`);
    if (lhs.shape.length > rhs.shape.length)
      rhs = unsqueeze(rhs, [...Array(lhs.shape.length - rhs.shape.length).keys()]);

    lhs = canonicalizeLayout(lhs);
    rhs = canonicalizeLayout(rhs);

    // Refine mode_shape of lhs
    const refinedMs = [];
    for (let d = 0; d < lhs.groupedModes.length; d++) {
      const lhsGroup = lhs.groupedModes[d];
      const rhsGroup = rhs.groupedModes[d];
      if (lhsGroup.length < rhsGroup.length)
        throw new Error(`Cannot divide layouts`);
      const refined = lhsGroup
        .slice(0, lhsGroup.length - rhsGroup.length)
        .map((m) => lhs.modeShape[m]);
      for (let i = 0; i < rhsGroup.length; i++) {
        const p = lhs.modeShape[lhsGroup[lhsGroup.length - rhsGroup.length + i]];
        const q = rhs.modeShape[rhsGroup[i]];
        if (p === q) {
          refined.push(p);
        } else if (i === 0 && p % q === 0) {
          refined.push(p / q);
          refined.push(q);
        } else {
          throw new Error(`Cannot divide layouts`);
        }
      }
      refinedMs.push(...refined);
    }
    lhs = _layoutWithModeShape(lhs, refinedMs);

    // Build result
    const shape = lhs.shape.map((a, i) => a / rhs.shape[i]);
    const ms = [];
    for (let d = 0; d < lhs.groupedModes.length; d++) {
      const lg = lhs.groupedModes[d];
      const rg = rhs.groupedModes[d];
      const pruned = rg.length > 0 ? lg.slice(0, -rg.length) : lg;
      for (const m of pruned) ms.push(lhs.modeShape[m]);
    }

    const modeMap2 = {};
    let idx = 0;
    for (let d = 0; d < lhs.groupedModes.length; d++) {
      const lg = lhs.groupedModes[d];
      const rg = rhs.groupedModes[d];
      const pruned = rg.length > 0 ? lg.slice(0, -rg.length) : lg;
      for (const m of pruned) {
        modeMap2[m] = idx++;
      }
    }

    const prunedSpatial =
      rhs.spatialModes.length > 0
        ? lhs.spatialModes.slice(0, -rhs.spatialModes.length)
        : lhs.spatialModes;
    const spatialModes = prunedSpatial.map((m) =>
      m >= 0 ? modeMap2[m] : m
    );
    const prunedLocal =
      rhs.localModes.length > 0
        ? lhs.localModes.slice(0, -rhs.localModes.length)
        : lhs.localModes;
    const localModes = prunedLocal.map((m) => modeMap2[m]);

    return registerLayout(shape, ms, spatialModes, localModes);
  }

  // auto_local_spatial
  function autoLocalSpatial(numThreads, shape) {
    const size = prod(shape);
    if (size % numThreads !== 0 && numThreads % size !== 0)
      throw new Error(`Cannot auto layout with shape [${shape}] and ${numThreads} threads`);

    const remainShape = [...shape];
    let remainThreads = numThreads;
    const spatialShape = new Array(shape.length).fill(1);

    for (let i = shape.length - 1; i >= 0; i--) {
      spatialShape[i] = gcd(remainThreads, remainShape[i]);
      remainThreads /= spatialShape[i];
      remainShape[i] /= spatialShape[i];
    }

    let ret = compose(local(...remainShape), spatial(...spatialShape));

    if (remainThreads !== 1) {
      ret = registerLayout(
        ret.shape,
        ret.modeShape,
        [-remainThreads, ...ret.spatialModes],
        ret.localModes
      );
    }
    return ret;
  }

  // ============================================================
  // Expression parser — parses layout DSL expressions
  // ============================================================

  function parseExpression(input) {
    const tokens = tokenize(input);
    const parser = new Parser(tokens);
    const result = parser.parseExpr();
    if (parser.pos < tokens.length) {
      throw new Error(`Unexpected token: ${tokens[parser.pos].value}`);
    }
    return result;
  }

  function tokenize(input) {
    const tokens = [];
    let i = 0;
    while (i < input.length) {
      if (/\s/.test(input[i])) {
        i++;
        continue;
      }
      if (/[a-zA-Z_]/.test(input[i])) {
        let start = i;
        while (i < input.length && /[a-zA-Z0-9_]/.test(input[i])) i++;
        tokens.push({ type: "ident", value: input.slice(start, i) });
        continue;
      }
      if (/[0-9]/.test(input[i])) {
        let start = i;
        while (i < input.length && /[0-9]/.test(input[i])) i++;
        tokens.push({ type: "number", value: parseInt(input.slice(start, i)) });
        continue;
      }
      if ("(),.*=/[]".includes(input[i])) {
        tokens.push({ type: input[i], value: input[i] });
        i++;
        continue;
      }
      throw new Error(`Unexpected character: '${input[i]}'`);
    }
    return tokens;
  }

  class Parser {
    constructor(tokens) {
      this.tokens = tokens;
      this.pos = 0;
    }

    peek() {
      return this.pos < this.tokens.length ? this.tokens[this.pos] : null;
    }

    consume(type) {
      const tok = this.peek();
      if (!tok || tok.type !== type)
        throw new Error(
          `Expected '${type}' but got ${tok ? `'${tok.value}'` : "end of input"}`
        );
      this.pos++;
      return tok;
    }

    parseExpr() {
      let left = this.parsePrimary();
      // Handle chained method calls: .spatial(2,3).local(4)
      while (this.peek() && this.peek().type === ".") {
        this.consume(".");
        const method = this.consume("ident").value;
        this.consume("(");
        const args = this.parseArgList();
        this.consume(")");
        left = this.applyMethod(left, method, args);
      }
      // Handle * and / operators
      while (
        this.peek() &&
        (this.peek().type === "*" || this.peek().type === "/")
      ) {
        const op = this.consume(this.peek().type).value;
        let right = this.parsePrimary();
        // right side can also have method chains
        while (this.peek() && this.peek().type === ".") {
          this.consume(".");
          const method = this.consume("ident").value;
          this.consume("(");
          const args = this.parseArgList();
          this.consume(")");
          right = this.applyMethod(right, method, args);
        }
        if (op === "*") {
          left = compose(left, right);
        } else {
          left = divide(left, right);
        }
      }
      return left;
    }

    parsePrimary() {
      const tok = this.peek();
      if (!tok) throw new Error("Unexpected end of input");

      if (tok.type === "(") {
        this.consume("(");
        const expr = this.parseExpr();
        this.consume(")");
        return expr;
      }

      if (tok.type === "ident") {
        const name = this.consume("ident").value;

        // Check if it's a function call
        if (this.peek() && this.peek().type === "(") {
          this.consume("(");
          const args = this.parseArgList();
          this.consume(")");
          return this.applyFunction(name, args);
        }

        throw new Error(`Unknown identifier: ${name}`);
      }

      throw new Error(`Unexpected token: ${tok.value}`);
    }

    parseArgList() {
      const args = [];
      let hasKeyword = false;
      if (this.peek() && this.peek().type !== ")") {
        args.push(this.parseArg());
        if (typeof args[0] === "object" && args[0]._keyword) hasKeyword = true;
        while (this.peek() && this.peek().type === ",") {
          this.consume(",");
          if (this.peek() && this.peek().type === ")") break;
          args.push(this.parseArg());
        }
      }
      return args;
    }

    parseArg() {
      // Check for keyword argument: name=value
      if (
        this.peek() &&
        this.peek().type === "ident" &&
        this.pos + 1 < this.tokens.length &&
        this.tokens[this.pos + 1].type === "="
      ) {
        const name = this.consume("ident").value;
        this.consume("=");
        const value = this.parseArgValue();
        return { _keyword: true, name, value };
      }
      return this.parseArgValue();
    }

    parseArgValue() {
      const tok = this.peek();
      if (!tok) throw new Error("Unexpected end of input");
      if (tok.type === "number") {
        this.consume("number");
        return tok.value;
      }
      if (tok.type === "[") {
        return this.parseList();
      }
      if (tok.type === "ident") {
        // Could be True/False or nested function
        if (tok.value === "True") { this.consume("ident"); return true; }
        if (tok.value === "False") { this.consume("ident"); return false; }
        // nested expression
        return this.parseExpr();
      }
      if (tok.type === "(") {
        return this.parseExpr();
      }
      throw new Error(`Unexpected in argument: ${tok.value}`);
    }

    parseList() {
      this.consume("[");
      const items = [];
      if (this.peek() && this.peek().type !== "]") {
        items.push(this.parseArgValue());
        while (this.peek() && this.peek().type === ",") {
          this.consume(",");
          if (this.peek() && this.peek().type === "]") break;
          items.push(this.parseArgValue());
        }
      }
      this.consume("]");
      return items;
    }

    extractPositionalAndKeyword(args) {
      const positional = [];
      const keyword = {};
      for (const a of args) {
        if (typeof a === "object" && a !== null && a._keyword) {
          keyword[a.name] = a.value;
        } else {
          positional.push(a);
        }
      }
      return { positional, keyword };
    }

    applyFunction(name, args) {
      const { positional, keyword } = this.extractPositionalAndKeyword(args);
      switch (name) {
        case "spatial":
          return spatial({ shape: positional, ranks: keyword.ranks || null });
        case "local":
          return local({ shape: positional, ranks: keyword.ranks || null });
        case "column_spatial":
          return columnSpatial(...positional);
        case "column_local":
          return columnLocal(...positional);
        case "replicated":
          return replicated(positional, keyword.num_workers);
        case "compose":
        case "product":
          if (positional.length !== 2)
            throw new Error(`${name}() requires exactly 2 arguments`);
          return compose(positional[0], positional[1]);
        case "divide":
          if (positional.length !== 2)
            throw new Error("divide() requires exactly 2 arguments");
          return divide(positional[0], positional[1]);
        case "reduce":
          if (positional.length < 2)
            throw new Error("reduce() requires layout and dims");
          return reduce(positional[0], positional[1], keyword.keepdims || false);
        case "permute":
          if (positional.length !== 2)
            throw new Error("permute() requires layout and dims");
          return permute(positional[0], positional[1]);
        case "reshape":
          if (positional.length !== 2)
            throw new Error("reshape() requires layout and shape");
          return reshape(positional[0], positional[1]);
        case "squeeze":
          if (positional.length !== 2)
            throw new Error("squeeze() requires layout and dims");
          return squeeze(positional[0], positional[1]);
        case "unsqueeze":
          if (positional.length !== 2)
            throw new Error("unsqueeze() requires layout and dims");
          return unsqueeze(positional[0], positional[1]);
        case "flatten":
          return flatten(positional[0], positional[1] || 0, positional[2] || -1);
        case "auto_local_spatial":
          return autoLocalSpatial(positional[0], positional[1]);
        case "register_layout":
          // register_layout(shape, mode_shape, spatial_modes, local_modes)
          if (positional.length !== 4)
            throw new Error("register_layout() requires 4 arguments: shape, mode_shape, spatial_modes, local_modes");
          return registerLayout(positional[0], positional[1], positional[2], positional[3]);
        default:
          throw new Error(`Unknown function: ${name}`);
      }
    }

    applyMethod(layout, method, args) {
      const { positional, keyword } = this.extractPositionalAndKeyword(args);
      switch (method) {
        case "spatial":
          return compose(layout, spatial({ shape: positional, ranks: keyword.ranks || null }));
        case "local":
          return compose(layout, local({ shape: positional, ranks: keyword.ranks || null }));
        case "column_spatial":
          return compose(layout, columnSpatial(...positional));
        case "column_local":
          return compose(layout, columnLocal(...positional));
        case "reduce_to":
          // reduce_to just reduces dims
          throw new Error("reduce_to is not supported in the interactive demo");
        default:
          throw new Error(`Unknown method: .${method}()`);
      }
    }
  }

  // ============================================================
  // Color generation — distinct colors for thread IDs
  // ============================================================

  function generateColors(n) {
    if (n <= 0) return [];
    const colors = [];
    // Use a perceptually-spaced palette via HSL
    const saturation = 70;
    const lightness = 82;
    for (let i = 0; i < n; i++) {
      const hue = (i * 360 / n + 15) % 360;
      colors.push(`hsl(${hue}, ${saturation}%, ${lightness}%)`);
    }
    return colors;
  }

  function generateBorderColors(n) {
    if (n <= 0) return [];
    const colors = [];
    const saturation = 70;
    const lightness = 45;
    for (let i = 0; i < n; i++) {
      const hue = (i * 360 / n + 15) % 360;
      colors.push(`hsl(${hue}, ${saturation}%, ${lightness}%)`);
    }
    return colors;
  }

  // ============================================================
  // Visualization renderer
  // ============================================================

  const MAX_GRID_SIZE = 4096; // refuse to render > this many cells

  function renderLayout(layout, container, infoDiv, mappingDiv) {
    container.innerHTML = "";

    let shape = [...layout.shape];

    // Normalize shape to 2D for display (like Python visualize_layout)
    if (shape.length > 3) {
      shape = shape.filter((s) => s > 1);
      while (shape.length > 3) {
        shape = [shape[0] * shape[1], ...shape.slice(2)];
      }
    }
    while (shape.length < 2) shape.unshift(1);

    // For 3D, we show batches separately
    let batches, rows, cols;
    if (shape.length === 3) {
      [batches, rows, cols] = shape;
    } else {
      batches = 1;
      [rows, cols] = shape;
    }

    if (rows * cols * batches > MAX_GRID_SIZE) {
      const msg = document.createElement("div");
      msg.className = "layout-demo-error";
      msg.textContent = `Layout too large to visualize (${rows}×${cols}×${batches} = ${rows * cols * batches} cells, max ${MAX_GRID_SIZE})`;
      container.appendChild(msg);
      return;
    }

    const displayLayout = layout.withShape(
      shape.length === 3 ? [batches, rows, cols] : [rows, cols]
    );

    // Collect all thread IDs for color mapping
    const allThreadIds = new Set();
    for (let b = 0; b < batches; b++) {
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          const globalIdx =
            shape.length === 3 ? [b, i, j] : [i, j];
          const threads = displayLayout.getSpatial(globalIdx);
          for (const t of threads) allThreadIds.add(t);
        }
      }
    }
    const sortedThreads = [...allThreadIds].sort((a, b) => a - b);
    const numThreads = sortedThreads.length;
    const bgColors = generateColors(numThreads);
    const borderColors = generateBorderColors(numThreads);
    const threadColorMap = {};
    sortedThreads.forEach((t, i) => {
      threadColorMap[t] = { bg: bgColors[i], border: borderColors[i] };
    });

    // Render info bar (into the external infoDiv)
    if (infoDiv) {
      infoDiv.innerHTML = `<span class="info-label">Threads:</span> <span class="info-value">${numThreads}</span>` +
        `<span class="info-sep">|</span>` +
        `<span class="info-label">Elements/thread:</span> <span class="info-value">${displayLayout.localSize}</span>` +
        `<span class="info-sep">|</span>` +
        `<span class="info-label">Tensor shape:</span> <span class="info-value">[${shape}]</span>`;
    }

    // Helper: render mapping steps into the external mappingDiv
    function showMapping(globalIdx) {
      if (!mappingDiv) return;
      const steps = displayLayout.getMappingSteps(globalIdx);
      const m = steps;
      const spatialMulti = m.spatialIndices.map((v, i) =>
        v === "*" ? `*` : `${v}`
      ).join(", ");
      const localMulti = m.localIndices.join(", ");
      const spatialFlatStr = m.spatialFlat.length === 1
        ? `${m.spatialFlat[0]}`
        : `[${m.spatialFlat.join(", ")}]`;
      mappingDiv.innerHTML =
        `<span class="mapping-step"><span class="mapping-label">Index</span> [${m.globalIndices.join(", ")}]</span>` +
        `<span class="mapping-arrow">&rarr;</span>` +
        `<span class="mapping-step"><span class="mapping-label">Mode Index</span> [${m.modeIndices.join(", ")}]</span>` +
        `<span class="mapping-arrow">&rarr;</span>` +
        `<span class="mapping-step"><span class="mapping-label">Spatial Index</span> (${spatialMulti})` +
        ` <span class="mapping-label">&amp; Local Index</span> (${localMulti})</span>` +
        `<span class="mapping-arrow">&rarr;</span>` +
        `<span class="mapping-step"><span class="mapping-label">Thread</span> ${spatialFlatStr}` +
        ` <span class="mapping-label">&amp; Local</span> ${m.localFlat}</span>`;
    }
    function clearMapping() {
      if (mappingDiv) mappingDiv.innerHTML = "&nbsp;";
    }

    // Track highlighted thread for legend clicks
    let highlightedThread = null;

    function updateHighlight() {
      const cells = container.querySelectorAll(".layout-cell");
      cells.forEach((cell) => {
        const threads = JSON.parse(cell.dataset.threads);
        if (highlightedThread !== null && !threads.includes(highlightedThread)) {
          cell.classList.add("dimmed");
        } else {
          cell.classList.remove("dimmed");
        }
      });
    }

    for (let b = 0; b < batches; b++) {
      if (batches > 1) {
        const batchLabel = document.createElement("div");
        batchLabel.className = "layout-demo-batch-label";
        batchLabel.textContent = `Batch ${b}`;
        container.appendChild(batchLabel);
      }

      const gridWrapper = document.createElement("div");
      gridWrapper.className = "layout-demo-grid-wrapper";

      const table = document.createElement("table");
      table.className = "layout-demo-grid";

      // Column header
      const thead = document.createElement("thead");
      const headerRow = document.createElement("tr");
      const cornerTh = document.createElement("th");
      cornerTh.className = "layout-corner";
      headerRow.appendChild(cornerTh);
      for (let j = 0; j < cols; j++) {
        const th = document.createElement("th");
        th.className = "layout-col-header";
        th.textContent = j;
        headerRow.appendChild(th);
      }
      thead.appendChild(headerRow);
      table.appendChild(thead);

      const tbody = document.createElement("tbody");
      for (let i = 0; i < rows; i++) {
        const tr = document.createElement("tr");

        // Row header
        const rowTh = document.createElement("th");
        rowTh.className = "layout-row-header";
        rowTh.textContent = i;
        tr.appendChild(rowTh);

        for (let j = 0; j < cols; j++) {
          const td = document.createElement("td");
          td.className = "layout-cell";

          const globalIdx =
            shape.length === 3 ? [b, i, j] : [i, j];
          const threads = displayLayout.getSpatial(globalIdx);
          const localIdx = displayLayout.getLocal(globalIdx);

          td.dataset.threads = JSON.stringify(threads);
          td.dataset.local = localIdx;
          td.dataset.row = i;
          td.dataset.col = j;
          td.dataset.batch = b;

          // Color by first thread (primary owner)
          const primaryThread = threads[0];
          const color = threadColorMap[primaryThread];
          td.style.backgroundColor = color.bg;
          td.style.borderColor = color.border;

          if (threads.length === 1) {
            td.innerHTML = `<span class="cell-thread">T${threads[0]}</span><span class="cell-local">${localIdx}</span>`;
          } else {
            // Replicated - show striped indicator
            td.classList.add("replicated");
            const stripes = threads
              .slice(0, 3)
              .map((t) => threadColorMap[t].bg)
              .join(", ");
            if (threads.length <= 3) {
              td.style.background = `linear-gradient(135deg, ${stripes})`;
            } else {
              td.style.background = `linear-gradient(135deg, ${stripes}, ${threadColorMap[threads[3]].bg})`;
            }
            td.innerHTML = `<span class="cell-thread">T${threads[0]}+${threads.length - 1}</span><span class="cell-local">${localIdx}</span>`;
          }

          // Hover handlers
          td.addEventListener("mouseenter", () => {
            const t = threads[0];
            showMapping(globalIdx);

            // Highlight all cells with same primary thread
            if (highlightedThread === null) {
              const allCells = container.querySelectorAll(".layout-cell");
              allCells.forEach((c) => {
                const ct = JSON.parse(c.dataset.threads);
                if (!ct.includes(t)) {
                  c.classList.add("dimmed");
                }
              });
            }
          });
          td.addEventListener("mouseleave", () => {
            clearMapping();
            if (highlightedThread === null) {
              const allCells = container.querySelectorAll(".layout-cell");
              allCells.forEach((c) => c.classList.remove("dimmed"));
            }
          });

          tr.appendChild(td);
        }
        tbody.appendChild(tr);
      }
      table.appendChild(tbody);
      gridWrapper.appendChild(table);
      container.appendChild(gridWrapper);
    }

    // Thread legend
    if (numThreads <= 128) {
      const legend = document.createElement("div");
      legend.className = "layout-demo-legend";
      const legendTitle = document.createElement("div");
      legendTitle.className = "legend-title";
      legendTitle.textContent = "Thread Legend (click to highlight)";
      legend.appendChild(legendTitle);

      const legendGrid = document.createElement("div");
      legendGrid.className = "legend-grid";

      sortedThreads.forEach((t) => {
        const item = document.createElement("div");
        item.className = "legend-item";
        item.dataset.thread = t;

        const swatch = document.createElement("span");
        swatch.className = "legend-swatch";
        swatch.style.backgroundColor = threadColorMap[t].bg;
        swatch.style.borderColor = threadColorMap[t].border;

        const label = document.createElement("span");
        label.className = "legend-label";
        label.textContent = `T${t}`;

        item.appendChild(swatch);
        item.appendChild(label);

        item.addEventListener("click", () => {
          if (highlightedThread === t) {
            highlightedThread = null;
            item.classList.remove("active");
          } else {
            highlightedThread = t;
            legend
              .querySelectorAll(".legend-item")
              .forEach((li) => li.classList.remove("active"));
            item.classList.add("active");
          }
          updateHighlight();
        });

        legendGrid.appendChild(item);
      });

      legend.appendChild(legendGrid);
      container.appendChild(legend);
    }
  }

  // ============================================================
  // Product Explorer — three-panel view for A * B = Result
  // ============================================================

  /** Normalize a layout shape to 2D for display. Returns [rows, cols]. */
  function normalizeShape2D(shape) {
    let s = [...shape];
    if (s.length > 2) {
      s = s.filter((v) => v > 1);
      while (s.length > 2) s = [s[0] * s[1], ...s.slice(2)];
    }
    while (s.length < 2) s.unshift(1);
    return s;
  }

  /**
   * Render a compact grid for an operand layout (no legend, no hover detail).
   * Returns the table element. Cells get data attributes for interaction.
   */
  function renderCompactGrid(layout, threadColorMap, label) {
    const wrapper = document.createElement("div");
    wrapper.className = "product-panel";

    const heading = document.createElement("div");
    heading.className = "product-panel-heading";
    heading.textContent = label;
    wrapper.appendChild(heading);

    const repr = document.createElement("div");
    repr.className = "product-panel-repr";
    repr.textContent = layout.toString();
    wrapper.appendChild(repr);

    const [rows, cols] = normalizeShape2D(layout.shape);
    const displayLayout = layout.withShape([rows, cols]);

    const gridWrap = document.createElement("div");
    gridWrap.className = "layout-demo-grid-wrapper";

    const table = document.createElement("table");
    table.className = "layout-demo-grid compact-grid";

    const tbody = document.createElement("tbody");
    for (let i = 0; i < rows; i++) {
      const tr = document.createElement("tr");
      for (let j = 0; j < cols; j++) {
        const td = document.createElement("td");
        td.className = "layout-cell";
        const threads = displayLayout.getSpatial([i, j]);
        const localIdx = displayLayout.getLocal([i, j]);

        td.dataset.threads = JSON.stringify(threads);
        td.dataset.local = localIdx;
        td.dataset.row = i;
        td.dataset.col = j;

        const primaryThread = threads[0];
        if (threadColorMap[primaryThread]) {
          const color = threadColorMap[primaryThread];
          td.style.backgroundColor = color.bg;
          td.style.borderColor = color.border;
        }

        if (threads.length === 1) {
          td.innerHTML = `<span class="cell-thread">T${threads[0]}</span><span class="cell-local">${localIdx}</span>`;
        } else {
          td.classList.add("replicated");
          td.innerHTML = `<span class="cell-thread">T${threads[0]}+${threads.length - 1}</span><span class="cell-local">${localIdx}</span>`;
        }

        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    gridWrap.appendChild(table);
    wrapper.appendChild(gridWrap);

    return { el: wrapper, displayLayout, rows, cols };
  }

  /**
   * Render the product explorer: outer panel, inner panel, result panel with tile borders.
   */
  function renderProductExplorer(layout, container, infoDiv, mappingDiv, showOperands, onToggleOperands) {
    container.innerHTML = "";

    const outer = layout._productOf.outer;
    const inner = layout._productOf.inner;

    // Build a combined thread color map from the result layout
    const resultShape = normalizeShape2D(layout.shape);
    const [rRows, rCols] = resultShape;

    if (rRows * rCols > MAX_GRID_SIZE) {
      const msg = document.createElement("div");
      msg.className = "layout-demo-error";
      msg.textContent = `Layout too large to visualize (${rRows}x${rCols} = ${rRows * rCols} cells, max ${MAX_GRID_SIZE})`;
      container.appendChild(msg);
      return;
    }

    const resultDisplay = layout.withShape(resultShape);

    // Collect all thread IDs from the result for a unified color map
    const allThreadIds = new Set();
    for (let i = 0; i < rRows; i++) {
      for (let j = 0; j < rCols; j++) {
        for (const t of resultDisplay.getSpatial([i, j])) allThreadIds.add(t);
      }
    }
    const sortedThreads = [...allThreadIds].sort((a, b) => a - b);
    const numThreads = sortedThreads.length;
    const bgColors = generateColors(numThreads);
    const borderColors = generateBorderColors(numThreads);
    const threadColorMap = {};
    sortedThreads.forEach((t, i) => {
      threadColorMap[t] = { bg: bgColors[i], border: borderColors[i] };
    });

    // Info bar
    if (infoDiv) {
      infoDiv.innerHTML =
        `<span class="info-label">Threads:</span> <span class="info-value">${numThreads}</span>` +
        `<span class="info-sep">|</span>` +
        `<span class="info-label">Elements/thread:</span> <span class="info-value">${resultDisplay.localSize}</span>` +
        `<span class="info-sep">|</span>` +
        `<span class="info-label">Result shape:</span> <span class="info-value">[${resultShape}]</span>`;
    }

    // Mapping div helpers
    function showMapping(globalIdx) {
      if (!mappingDiv) return;
      const steps = resultDisplay.getMappingSteps(globalIdx);
      const m = steps;
      const spatialMulti = m.spatialIndices.map((v) => v === "*" ? "*" : `${v}`).join(", ");
      const localMulti = m.localIndices.join(", ");
      const spatialFlatStr = m.spatialFlat.length === 1 ? `${m.spatialFlat[0]}` : `[${m.spatialFlat.join(", ")}]`;
      mappingDiv.innerHTML =
        `<span class="mapping-step"><span class="mapping-label">Index</span> [${m.globalIndices.join(", ")}]</span>` +
        `<span class="mapping-arrow">&rarr;</span>` +
        `<span class="mapping-step"><span class="mapping-label">Mode Index</span> [${m.modeIndices.join(", ")}]</span>` +
        `<span class="mapping-arrow">&rarr;</span>` +
        `<span class="mapping-step"><span class="mapping-label">Spatial Index</span> (${spatialMulti})` +
        ` <span class="mapping-label">&amp; Local Index</span> (${localMulti})</span>` +
        `<span class="mapping-arrow">&rarr;</span>` +
        `<span class="mapping-step"><span class="mapping-label">Thread</span> ${spatialFlatStr}` +
        ` <span class="mapping-label">&amp; Local</span> ${m.localFlat}</span>`;
    }
    function clearMapping() {
      if (mappingDiv) mappingDiv.innerHTML = "&nbsp;";
    }

    // --- Toggle button ---
    const toggleBtn = document.createElement("button");
    toggleBtn.className = "product-toggle-btn";
    toggleBtn.textContent = showOperands ? "Hide operands (A, B)" : "Show operands (A, B)";
    toggleBtn.addEventListener("click", () => {
      onToggleOperands(!showOperands);
    });
    container.appendChild(toggleBtn);

    // --- Operand panels (conditionally shown) ---
    const outerPanel = renderCompactGrid(outer, threadColorMap, "A (outer)");
    const innerPanel = renderCompactGrid(inner, threadColorMap, "B (inner)");

    if (showOperands) {
      const operandRow = document.createElement("div");
      operandRow.className = "product-operand-row";

      const equalsLabel = document.createElement("div");
      equalsLabel.className = "product-equals-label";

      operandRow.appendChild(outerPanel.el);
      operandRow.appendChild(equalsLabel);
      operandRow.appendChild(innerPanel.el);
      container.appendChild(operandRow);
    }

    // --- Result panel with tile borders ---
    const resultHeading = document.createElement("div");
    resultHeading.className = "product-result-heading";
    if (showOperands) {
      resultHeading.textContent = "A \u00D7 B (result) \u2014 each element of A is replaced by a tile shaped like B";
    } else {
      resultHeading.textContent = "Result";
    }
    container.appendChild(resultHeading);

    const gridWrap = document.createElement("div");
    gridWrap.className = "layout-demo-grid-wrapper";

    const table = document.createElement("table");
    table.className = "layout-demo-grid result-grid";

    // Tile dimensions: inner layout shape determines tile size
    const [innerRows, innerCols] = normalizeShape2D(inner.shape);

    let highlightedThread = null;

    function updateHighlight() {
      container.querySelectorAll(".layout-cell").forEach((cell) => {
        const threads = JSON.parse(cell.dataset.threads);
        if (highlightedThread !== null && !threads.includes(highlightedThread)) {
          cell.classList.add("dimmed");
        } else {
          cell.classList.remove("dimmed");
        }
      });
    }

    // Column header
    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");
    const cornerTh = document.createElement("th");
    cornerTh.className = "layout-corner";
    headerRow.appendChild(cornerTh);
    for (let j = 0; j < rCols; j++) {
      const th = document.createElement("th");
      th.className = "layout-col-header";
      if (j % innerCols === 0 && j > 0) th.classList.add("tile-border-left");
      th.textContent = j;
      headerRow.appendChild(th);
    }
    thead.appendChild(headerRow);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    for (let i = 0; i < rRows; i++) {
      const tr = document.createElement("tr");

      // Row header
      const rowTh = document.createElement("th");
      rowTh.className = "layout-row-header";
      if (i % innerRows === 0 && i > 0) rowTh.classList.add("tile-border-top");
      rowTh.textContent = i;
      tr.appendChild(rowTh);

      for (let j = 0; j < rCols; j++) {
        const td = document.createElement("td");
        td.className = "layout-cell";

        // Tile border classes
        if (i % innerRows === 0 && i > 0) td.classList.add("tile-border-top");
        if (j % innerCols === 0 && j > 0) td.classList.add("tile-border-left");

        // Which outer tile does this cell belong to?
        const outerRow = Math.floor(i / innerRows);
        const outerCol = Math.floor(j / innerCols);
        td.dataset.tileRow = outerRow;
        td.dataset.tileCol = outerCol;

        const globalIdx = [i, j];
        const threads = resultDisplay.getSpatial(globalIdx);
        const localIdx = resultDisplay.getLocal(globalIdx);

        td.dataset.threads = JSON.stringify(threads);
        td.dataset.local = localIdx;
        td.dataset.row = i;
        td.dataset.col = j;

        const primaryThread = threads[0];
        const color = threadColorMap[primaryThread];
        if (color) {
          td.style.backgroundColor = color.bg;
          td.style.borderColor = color.border;
        }

        if (threads.length === 1) {
          td.innerHTML = `<span class="cell-thread">T${threads[0]}</span><span class="cell-local">${localIdx}</span>`;
        } else {
          td.classList.add("replicated");
          const stripes = threads.slice(0, 3).map((t) => threadColorMap[t].bg).join(", ");
          if (threads.length <= 3) {
            td.style.background = `linear-gradient(135deg, ${stripes})`;
          } else {
            td.style.background = `linear-gradient(135deg, ${stripes}, ${threadColorMap[threads[3]].bg})`;
          }
          td.innerHTML = `<span class="cell-thread">T${threads[0]}+${threads.length - 1}</span><span class="cell-local">${localIdx}</span>`;
        }

        // Hover: highlight tile + show mapping
        td.addEventListener("mouseenter", () => {
          showMapping(globalIdx);

          // Highlight the tile this cell belongs to
          if (highlightedThread === null) {
            container.querySelectorAll(".result-grid .layout-cell").forEach((c) => {
              if (c.dataset.tileRow !== td.dataset.tileRow || c.dataset.tileCol !== td.dataset.tileCol) {
                c.classList.add("dimmed");
              }
            });
            // Also highlight corresponding cell in outer panel
            outerPanel.el.querySelectorAll(".layout-cell").forEach((c) => {
              if (c.dataset.row !== td.dataset.tileRow || c.dataset.col !== td.dataset.tileCol) {
                c.classList.add("dimmed");
              }
            });
          }
        });
        td.addEventListener("mouseleave", () => {
          clearMapping();
          if (highlightedThread === null) {
            container.querySelectorAll(".layout-cell").forEach((c) => c.classList.remove("dimmed"));
          }
        });

        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    gridWrap.appendChild(table);
    container.appendChild(gridWrap);

    // Hover on outer panel: highlight corresponding tile in result
    outerPanel.el.querySelectorAll(".layout-cell").forEach((cell) => {
      cell.addEventListener("mouseenter", () => {
        const tr = cell.dataset.row;
        const tc = cell.dataset.col;
        container.querySelectorAll(".result-grid .layout-cell").forEach((c) => {
          if (c.dataset.tileRow !== tr || c.dataset.tileCol !== tc) {
            c.classList.add("dimmed");
          }
        });
        outerPanel.el.querySelectorAll(".layout-cell").forEach((c) => {
          if (c.dataset.row !== tr || c.dataset.col !== tc) c.classList.add("dimmed");
        });
      });
      cell.addEventListener("mouseleave", () => {
        container.querySelectorAll(".layout-cell").forEach((c) => c.classList.remove("dimmed"));
      });
    });

    // Hover on inner panel: highlight the same position in every tile
    innerPanel.el.querySelectorAll(".layout-cell").forEach((cell) => {
      cell.addEventListener("mouseenter", () => {
        const ir = parseInt(cell.dataset.row);
        const ic = parseInt(cell.dataset.col);
        container.querySelectorAll(".result-grid .layout-cell").forEach((c) => {
          const ri = parseInt(c.dataset.row);
          const ci = parseInt(c.dataset.col);
          if (ri % innerRows !== ir || ci % innerCols !== ic) {
            c.classList.add("dimmed");
          }
        });
        innerPanel.el.querySelectorAll(".layout-cell").forEach((c) => {
          if (c.dataset.row !== cell.dataset.row || c.dataset.col !== cell.dataset.col) {
            c.classList.add("dimmed");
          }
        });
      });
      cell.addEventListener("mouseleave", () => {
        container.querySelectorAll(".layout-cell").forEach((c) => c.classList.remove("dimmed"));
      });
    });

    // Thread legend
    if (numThreads <= 128) {
      const legend = document.createElement("div");
      legend.className = "layout-demo-legend";
      const legendTitle = document.createElement("div");
      legendTitle.className = "legend-title";
      legendTitle.textContent = "Thread Legend (click to highlight)";
      legend.appendChild(legendTitle);

      const legendGrid = document.createElement("div");
      legendGrid.className = "legend-grid";

      sortedThreads.forEach((t) => {
        const item = document.createElement("div");
        item.className = "legend-item";
        const swatch = document.createElement("span");
        swatch.className = "legend-swatch";
        swatch.style.backgroundColor = threadColorMap[t].bg;
        swatch.style.borderColor = threadColorMap[t].border;
        const label = document.createElement("span");
        label.className = "legend-label";
        label.textContent = `T${t}`;
        item.appendChild(swatch);
        item.appendChild(label);
        item.addEventListener("click", () => {
          if (highlightedThread === t) {
            highlightedThread = null;
            item.classList.remove("active");
          } else {
            highlightedThread = t;
            legend.querySelectorAll(".legend-item").forEach((li) => li.classList.remove("active"));
            item.classList.add("active");
          }
          updateHighlight();
        });
        legendGrid.appendChild(item);
      });

      legend.appendChild(legendGrid);
      container.appendChild(legend);
    }
  }

  // ============================================================
  // Presets
  // ============================================================

  const PRESET_CATEGORIES = [
    {
      name: "Basics",
      presets: [
        { label: "local(3, 4)", expr: "local(3, 4)" },
        { label: "spatial(3, 2)", expr: "spatial(3, 2)" },
        { label: "local(3, 4).spatial(2, 3)", expr: "local(3, 4).spatial(2, 3)" },
        { label: "spatial(2, 3).local(3, 4)", expr: "spatial(2, 3).local(3, 4)" },
        { label: "column_local(2, 3)", expr: "column_local(2, 3)" },
        { label: "column_spatial(2, 3)", expr: "column_spatial(2, 3)" },
        { label: "reduce(spatial(3, 4), [0])", expr: "reduce(spatial(3, 4), [0])" },
        { label: "spatial(4, 8).local(4, 4)", expr: "spatial(4, 8).local(4, 4)" },
        { label: "local(2, 2).spatial(4, 4).local(2, 2)", expr: "local(2, 2).spatial(4, 4).local(2, 2)" },
      ],
    },
    {
      name: "Ampere mma.sync",
      presets: [
        // m16n8k16 f16 (vec_k=1) — operand A
        { label: "m16n8k16 f16 A", expr: "column_local(2, 2).spatial(8, 4).local(1, 2)" },
        // m16n8k16 f16 (vec_k=1) — operand B
        { label: "m16n8k16 f16 B", expr: "local(2, 1).column_spatial(4, 8).local(2, 1)" },
        // m16n8k16 f16 (vec_k=1) — accumulator C
        { label: "m16n8k16 f16 C", expr: "local(2, 1).spatial(8, 4).local(1, 2)" },
        // m8n8k16 i8 (vec_k=1) — operand A
        { label: "m8n8k16 i8 A", expr: "spatial(8, 4).local(1, 4)" },
        // m8n8k16 i8 (vec_k=1) — operand B
        { label: "m8n8k16 i8 B", expr: "column_spatial(4, 8).local(4, 1)" },
        // m8n8k16 i8 (vec_k=1) — accumulator C
        { label: "m8n8k16 i8 C", expr: "spatial(8, 4).local(1, 2)" },
        // m16n8k32 i8 (vec_k=1) — operand A
        { label: "m16n8k32 i8 A", expr: "column_local(2, 2).spatial(8, 4).local(1, 4)" },
        // m16n8k32 i8 (vec_k=1) — operand B
        { label: "m16n8k32 i8 B", expr: "local(2, 1).column_spatial(4, 8).local(4, 1)" },
        // m16n8k32 i8 (vec_k=1) — accumulator C
        { label: "m16n8k32 i8 C", expr: "local(2, 1).spatial(8, 4).local(1, 2)" },
      ],
    },
    {
      name: "Hopper wgmma",
      presets: [
        // wgmma m64n8k16 f16 — output D
        { label: "m64n8k16 f16 D", expr: "column_spatial(4, 1).column_local(2, 1).spatial(8, 4).local(2)" },
        // wgmma m64n16k16 f16 — output D
        { label: "m64n16k16 f16 D", expr: "column_spatial(4, 1).column_local(2, 2).spatial(8, 4).local(2)" },
        // wgmma m64n32k16 f16 — output D
        { label: "m64n32k16 f16 D", expr: "column_spatial(4, 1).column_local(2, 4).spatial(8, 4).local(2)" },
        // wgmma m64n64k16 f16 — output D (large)
        { label: "m64n64k16 f16 D", expr: "column_spatial(4, 1).column_local(2, 8).spatial(8, 4).local(2)" },
      ],
    },
    {
      name: "Blackwell tcgen05 ld/st",
      presets: [
        // R32x32B
        { label: "32x32B", expr: "spatial(32, 1)" },
        // R16x64B — special interleaved layout
        { label: "16x64B", expr: "register_layout([16, 2], [2, 8, 2], [1, 2, 0], [])" },
        // R16x128B
        { label: "16x128B", expr: "local(2, 1).spatial(8, 4)" },
        // R16x256B
        { label: "16x256B", expr: "local(2, 1).spatial(8, 4).local(1, 2)" },
      ],
    },
  ];

  // ============================================================
  // Main initialization
  // ============================================================

  function initDemo() {
    const root = document.getElementById("layout-demo-root");
    if (!root) return;

    // Build the UI
    root.innerHTML = `
      <div class="layout-demo-container">
        <div class="layout-demo-input-panel">
          <div class="input-row">
            <label for="layout-expr">Layout expression:</label>
            <input type="text" id="layout-expr" spellcheck="false"
              placeholder="e.g. spatial(4, 8).local(2, 2)"
              value="local(3, 4).spatial(2, 3)" />
            <button id="layout-run" title="Evaluate">Run</button>
          </div>
          <div class="preset-row" id="preset-row"></div>
          <div id="layout-error" class="layout-demo-error" style="display:none"></div>
        </div>
        <div class="layout-demo-labeled-row">
          <span class="labeled-row-label">Canonical Form</span>
          <div class="layout-demo-repr" id="layout-repr"></div>
        </div>
        <div class="layout-demo-labeled-row">
          <span class="labeled-row-label">Statistics</span>
          <div class="layout-demo-info" id="layout-info"></div>
        </div>
        <div class="layout-demo-labeled-row">
          <span class="labeled-row-label">Mapping Process</span>
          <div class="layout-demo-mapping" id="layout-mapping">&nbsp;</div>
        </div>
        <div class="layout-demo-output" id="layout-output"></div>
      </div>
    `;

    const input = document.getElementById("layout-expr");
    const runBtn = document.getElementById("layout-run");
    const errorDiv = document.getElementById("layout-error");
    const reprDiv = document.getElementById("layout-repr");
    const infoDiv = document.getElementById("layout-info");
    const mappingDiv = document.getElementById("layout-mapping");
    const outputDiv = document.getElementById("layout-output");
    const presetRow = document.getElementById("preset-row");

    // Persistent state for operand visibility
    let showOperands = true;

    // Preset buttons — grouped by category
    PRESET_CATEGORIES.forEach((cat) => {
      const group = document.createElement("div");
      group.className = "preset-group";
      const label = document.createElement("span");
      label.className = "preset-label";
      label.textContent = cat.name + ":";
      group.appendChild(label);
      const buttons = document.createElement("div");
      buttons.className = "preset-buttons";
      cat.presets.forEach((p) => {
        const btn = document.createElement("button");
        btn.className = "preset-btn";
        btn.textContent = p.label;
        btn.addEventListener("click", () => {
          input.value = p.expr;
          evaluate();
        });
        buttons.appendChild(btn);
      });
      group.appendChild(buttons);
      presetRow.appendChild(group);
    });

    function evaluate() {
      errorDiv.style.display = "none";
      reprDiv.textContent = "";
      infoDiv.innerHTML = "";
      mappingDiv.innerHTML = "&nbsp;";
      try {
        const layout = parseExpression(input.value.trim());
        reprDiv.textContent = layout.toString();
        if (layout._productOf) {
          renderProductExplorer(layout, outputDiv, infoDiv, mappingDiv, showOperands, (newVal) => {
            showOperands = newVal;
            evaluate();
          });
        } else {
          renderLayout(layout, outputDiv, infoDiv, mappingDiv);
        }
      } catch (e) {
        errorDiv.style.display = "block";
        errorDiv.textContent = e.message;
        outputDiv.innerHTML = "";
      }
    }

    runBtn.addEventListener("click", evaluate);
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") evaluate();
    });

    // Initial render
    evaluate();
  }

  // Run on DOMContentLoaded
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initDemo);
  } else {
    initDemo();
  }
})();

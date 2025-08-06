# Tilus: A Tile-Level GPU Kernel Programming Language

**Tilus** is a powerful domain-specific language (DSL) for GPU programming that offers:

* **Thread-block-level granularity** with **tensors** as the primary data type.
* **Explicit control** over shared memory and register tensors (unlike Triton).
* **Low-precision types** with arbitrary bit-widths (1 to 8 bits).

It also includes automatic tuning, caching, and a Pythonic interface for ease of use.

Tilus is pronounced as tie-lus, /ˈtaɪləs/.

## Getting Started

### Installation
Install Tilus using `pip`:
```
pip install tilus
```

### Usage
To get started, refer to the [tutorials]() to learn how to program kernels with Tilus. You can also find more examples [here](https://github.com/NVIDIA/tilus/tree/main/examples).

## Research
This project is based on the following research paper:

```bibtex
@article{ding2025tilus,
  title={Tilus: A Virtual Machine for Arbitrary Low-Precision GPGPU Computation in LLM Serving},
  author={Ding, Yaoyao and Hou, Bohan and Zhang, Xiao and Lin, Allan and
    Chen, Tianqi and Hao, Cody Yu and Wang, Yida and Pekhimenko, Gennady},
  journal={arXiv preprint arXiv:2504.12984},
  year={2025}
}
```

## Acknowledgement
We would like to acknowledge the following projects for their influence on Tilus's design and development:
- **Hidet**: We build Tilus on top of the Hidet IR and runtime, which allows us to write a compiler in pure Python.
- **TVM**: Hidet's initial IR was adopted from TVM.
- **Triton**: We were inspired by its core idea of defining kernels at a thread-block granularity and operating on tiles.
- **Hexcute**: This project provided the idea of using automatic layout inference to simplify programming.

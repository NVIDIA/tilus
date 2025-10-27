# Tilus: A Tile-Level GPU Kernel Programming Language
[Documentation](https://nvidia.github.io/tilus/index.html) | [Paper](https://arxiv.org/abs/2504.12984)

**Tilus** is a powerful research domain-specific language (DSL) for GPU programming that offers:

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

> [!NOTE]  
> Tilus depends on `cuda-python`. If your GPU driver is older than **580.65.06**, you will need to install an older version of cuda-python to ensure compatibility.
> ```
> pip install tilus "cuda-python<13"
> ```

### Usage

To get started, refer to the [tutorials](https://nvidia.github.io/tilus/getting-started/tutorials/__init__.html) to learn how to program kernels with Tilus. 

You can also check more [examples](https://github.com/NVIDIA/tilus/tree/main/examples) of using Tilus.

You can learn more on different topics in the [programming guide](https://nvidia.github.io/tilus/programming-guides/overview.html).

## Research
This project is based on the following research paper:

```bibtex
@article{ding2025tilus,
  title={Tilus: A Tile-Level GPGPU Programming Language for Low-Precision Computation},
  author={Ding, Yaoyao and Hou, Bohan and Zhang, Xiao and Lin, Allan and
    Chen, Tianqi and Hao, Cody Yu and Wang, Yida and Pekhimenko, Gennady},
  journal={arXiv preprint arXiv:2504.12984},
  year={2025}
}
```

## Acknowledgement
We would like to acknowledge the following projects for their influence on Tilus's design and development:
- **Hidet**: We take Hidet IR as our low-level target and reuse its runtime system.
- **TVM**: Hidet's initial IR was adopted from TVM, and we also learned a lot from TVM on how to build a compiler.
- **Triton**: The core idea of defining kernels at a thread-block level and working with tiles was inspired by Triton. 
- **Hexcute**: We adopted the idea of using automatic layout inference to simplify programming from Hexcute.

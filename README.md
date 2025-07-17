# Tilus: A Domain-Specific Language for High-Performance GPU Programming

**Tilus** is a domain-specific language (DSL) for GPU programming, designed with:

* Thread-block-level granularity and tensors as the core data type
* Explicit control over shared memory and tensor layouts (unlike Triton)
* Support for low-precision types with arbitrary bit-widths

Additional features include automatic tuning, caching, and a Pythonic interface for ease of use.

Tilus is proununced as tie-lus, /ˈtaɪləs/.

Please cite the following paper if you use Tilus in your work:

```bibtex
@article{ding2025tilus,
  title={Tilus: A Virtual Machine for Arbitrary Low-Precision GPGPU Computation in LLM Serving},
  author={Ding, Yaoyao and Hou, Bohan and Zhang, Xiao and Lin, Allan and Chen, Tianqi and Hao, Cody Yu and Wang, Yida and Pekhimenko, Gennady},
  journal={arXiv preprint arXiv:2504.12984},
  year={2025}
}
```


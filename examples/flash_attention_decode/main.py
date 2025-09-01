import pandas as pd
import torch
from tilus.utils import benchmark_func
from tilus_kernel import (
    fused_gdn_gating_tilus,
    fused_recurrent_gated_delta_rule_update_fwd_tilus,
)
from torch_kernel import (
    fused_gdn_gating_torch,
    fused_recurrent_gated_delta_rule_update_fwd_torch,
)
from triton_kernel import (
    fused_gdn_gating,
    fused_recurrent_gated_delta_rule_update_fwd_triton,
)


def main_fused_gdn_gating():
    """Test and benchmark fused GDN gating implementations."""
    headers = ["name", "(batch, num_heads)", "latency (ms)"]
    rows = []

    for batch, num_heads in [
        [1, 8],
        [1, 16],
        [1, 32],
        [1, 64],
        [1, 128],
        [1, 256],
        [2, 128],
        [4, 128],
        [8, 128],
    ]:
        # Create test inputs
        A_log = torch.randn(num_heads, device="cuda", dtype=torch.float32) * 0.1
        a = torch.randn(batch, num_heads, device="cuda", dtype=torch.bfloat16)
        dt_bias = torch.randn(num_heads, device="cuda", dtype=torch.float32) * 0.1
        beta = 1.0
        threshold = 20.0

        arguments = {
            "A_log": A_log,
            "a": a,
            "dt_bias": dt_bias,
            "beta": beta,
            "threshold": threshold,
        }

        # Clone arguments for each implementation
        torch_arguments = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in arguments.items()
        }
        triton_arguments = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in arguments.items()
        }
        tilus_arguments = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in arguments.items()
        }

        # Test correctness
        torch_output = fused_gdn_gating_torch(**torch_arguments)
        triton_output = fused_gdn_gating(**triton_arguments)
        tilus_output = fused_gdn_gating_tilus(**tilus_arguments)

        # Verify outputs match
        torch.testing.assert_close(triton_output, torch_output, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(tilus_output, torch_output, atol=1e-3, rtol=1e-3)
        print(f"✓ Correctness test passed for batch={batch}, num_heads={num_heads}")

        # Benchmark all implementations
        for name, func, args in [
            ["torch", fused_gdn_gating_torch, torch_arguments],
            ["triton", fused_gdn_gating, triton_arguments],
            ["tilus", fused_gdn_gating_tilus, tilus_arguments],
        ]:
            latency = benchmark_func(lambda: func(**args), warmup=10, repeat=50)
            rows.append([name, f"({batch}, {num_heads})", f"{latency:.3f}"])

    # Print benchmark results
    df = pd.DataFrame(rows, columns=headers)
    print("\nFused GDN Gating Benchmark Results:")
    print(df)
    print()


def main_fused_recurrent_gated_delta_rule_update_fwd():
    headers = ["name", "(B, T, H, K)", "(HV, V)", "latency (ms)"]
    rows = []

    for B, T, H, K, HV, V in [
        [1, 1, 4, 128, 8, 128],
        [1, 2, 4, 128, 8, 128],
        [1, 4, 4, 128, 8, 128],
        [1, 8, 4, 128, 8, 128],
        [1, 16, 4, 128, 8, 128],
        [1, 32, 4, 128, 8, 128],
        [1, 64, 4, 128, 8, 128],
        [1, 128, 4, 128, 8, 128],
    ]:
        q = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, T, HV, V, device="cuda", dtype=torch.bfloat16)
        g = torch.randn(B, T, HV, device="cuda", dtype=torch.float32) * 0.1
        beta = torch.rand(B, T, HV, device="cuda", dtype=torch.bfloat16) * 0.5 + 0.5
        initial_state_source = (
            torch.randn(129, HV, K, V, device="cuda", dtype=torch.float32) * 0.1
        )
        initial_state_indices = 126 - torch.arange(T, device="cuda", dtype=torch.int32)
        scale = K**-0.5
        cu_seqlens = torch.arange(T + 1, device="cuda", dtype=torch.int32)

        arguments = {
            "q": q,
            "k": k,
            "v": v,
            "g": g,
            "beta": beta,
            "scale": scale,
            "initial_state_source": initial_state_source,
            "initial_state_indices": initial_state_indices,
            "use_qk_l2norm_in_kernel": True,
        }

        torch_arguments = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in arguments.items()
        }
        tilus_arguments = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in arguments.items()
        }
        triton_arguments = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in arguments.items()
        }
        triton_arguments["cu_seqlens"] = cu_seqlens

        expect_o = fused_recurrent_gated_delta_rule_update_fwd_torch(**torch_arguments)
        actual_o = fused_recurrent_gated_delta_rule_update_fwd_tilus(**tilus_arguments)
        triton_o = fused_recurrent_gated_delta_rule_update_fwd_triton(**triton_arguments)

        torch.testing.assert_close(actual_o, expect_o, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(triton_o, expect_o, atol=1e-3, rtol=1e-3)

        # benchmark
        for name, func, args in [
            [
                "triton",
                fused_recurrent_gated_delta_rule_update_fwd_triton,
                triton_arguments,
            ],
            ["tilus", fused_recurrent_gated_delta_rule_update_fwd_tilus, tilus_arguments],
        ]:
            latency = benchmark_func(lambda: func(**args), warmup=10, repeat=50)
            rows.append([name, f"({B}, {T}, {H}, {K})", f"({HV}, {V})", f"{latency:.3f}"])
        df = pd.DataFrame(rows, columns=headers)
        print(df)
        print()


def main():
    """Run all tests and benchmarks."""
    print("Running Fused GDN Gating tests and benchmarks...")
    main_fused_gdn_gating()

    print("Running Fused Recurrent Gated Delta Rule Update tests and benchmarks...")
    main_fused_recurrent_gated_delta_rule_update_fwd()


if __name__ == "__main__":
    main()

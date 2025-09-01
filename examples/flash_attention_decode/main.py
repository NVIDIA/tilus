import math
from typing import Optional

import pandas as pd
import tilus
import torch
from tilus import bfloat16, float32, int32
from tilus.utils import benchmark_func, cdiv
from torch_kernel import fused_recurrent_gated_delta_rule_update_fwd_torch
from triton_kernel import fused_recurrent_gated_delta_rule_update_fwd_triton


@tilus.autotune("num_warps", [1, 2, 4])
@tilus.autotune("BV", [2, 4, 8, 16, 32, 64, 128])
class FusedRecurrentGatedDeltaRuleUpdateFwdKernel(tilus.Script):
    def __init__(self, num_warps: int, BV: int):
        super().__init__()
        self.num_warps = num_warps
        self.BV = BV

    def __call__(
        self,
        q_ptr: ~bfloat16,
        k_ptr: ~bfloat16,
        v_ptr: ~bfloat16,
        g_ptr: ~float32,
        o_ptr: ~bfloat16,
        beta_ptr: ~bfloat16,
        scale: float,
        initial_state_source_ptr: ~float32,
        initial_state_indices_ptr: ~int32,
        T: int32,
        B: int,
        H: int,
        HV: int,
        K: int,
        V: int,
        MAX_T: int,
        USE_INITIAL_STATE: bool,
        USE_QK_L2NORM_IN_KERNEL: bool,
    ):
        self.attrs.warps = self.num_warps
        self.attrs.blocks = (T, cdiv(V, self.BV), HV)

        g_q = self.global_view(q_ptr, dtype=bfloat16, shape=[B, T, H, K])
        g_k = self.global_view(k_ptr, dtype=bfloat16, shape=[B, T, H, K])
        g_v = self.global_view(v_ptr, dtype=bfloat16, shape=[B, T, HV, V])
        g_g = self.global_view(g_ptr, dtype=float32, shape=[B, T, HV])
        g_o = self.global_view(o_ptr, dtype=bfloat16, shape=[B, T, HV, V])
        g_beta = self.global_view(beta_ptr, dtype=bfloat16, shape=[B, T, HV])
        initial_state_source = self.global_view(
            initial_state_source_ptr, dtype=float32, shape=[MAX_T, HV, K, V]
        )
        initial_state_indices = self.global_view(
            initial_state_indices_ptr, dtype=int32, shape=[T]
        )

        i_t, i_bv, i_hv = self.blockIdx
        i_h = i_hv // (HV // H)

        r_q = self.load_global(g_q, offsets=[0, i_t, i_h, 0], shape=[K], dims=[3]).to(
            float32
        )
        r_k = self.load_global(g_k, offsets=[0, i_t, i_h, 0], shape=[K], dims=[3]).to(
            float32
        )

        # self.print_tensor('r_q ', r_q)
        # self.print_tensor('r_k ', r_k)

        # normalize q and k
        if USE_QK_L2NORM_IN_KERNEL:
            r_q = r_q / self.sqrt(self.sum(r_q * r_q, dim=0))
            r_k = r_k / self.sqrt(self.sum(r_k * r_k, dim=0))

        r_q = r_q * scale

        # self.print_tensor('normalized r_q ', r_q)
        # self.print_tensor('normalized r_k ', r_k)

        # load initial state
        if USE_INITIAL_STATE:
            state_idx: int32 = initial_state_indices[i_t]
            r_h = self.load_global(
                initial_state_source,
                offsets=[state_idx, i_hv, 0, i_bv * self.BV],
                shape=[K, self.BV],
                dims=[2, 3],
            )
        else:
            state_idx: int32 = -1
            r_h = self.register_tensor(dtype=float32, shape=[K, self.BV], init=0.0)

        # H' = alpha * H : [K, BV] = [] * [K, BV]
        alpha: float32 = math.exp(g_g[0, i_t, i_hv])
        r_h = r_h * alpha

        # p = k * H : [BV] = reduce([K, 1] * [K, BV], dim=0)
        r_p = self.sum(self.unsqueeze(r_k, dim=1) * r_h, dim=0)
        # self.print_tensor('r_p ', r_p)

        # r = beta * (v - p) : [BV] = [] * ([BV] - [BV])
        beta: float32 = float32(g_beta[0, i_t, i_hv])
        r_v = self.load_global(
            g_v, offsets=[0, i_t, i_hv, i_bv * self.BV], shape=[self.BV], dims=[3]
        ).to(float32)
        r_r = beta * (r_v - r_p)

        # H'' = H' + k * r : [K, BV] = [K, BV] + [K] * [BV]
        r_h += self.unsqueeze(r_k, dim=1) * self.unsqueeze(r_r, dim=0)

        # o = q * h : [BV] = [K] * [K, BV]
        r_o = self.sum(self.unsqueeze(r_q, dim=1) * r_h, dim=0).to(bfloat16)

        self.store_global(g_o, r_o, offsets=[0, i_t, i_hv, i_bv * self.BV], dims=[3])
        if state_idx >= 0:
            self.store_global(
                initial_state_source,
                r_h,
                offsets=[state_idx, i_hv, 0, i_bv * self.BV],
                dims=[2, 3],
            )

        self.annotate_layout(
            r_h,
            self.cuda.default_register_layout(
                num_warps=self.num_warps, dtype=float32, shape=[K, self.BV]
            ),
        )


def fused_recurrent_gated_delta_rule_update_fwd_tilus(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
) -> torch.Tensor:
    _ = cu_seqlens
    o = torch.empty_like(v)

    B, T, H, K = q.shape
    HV, V = v.shape[-2:]
    MAX_T = initial_state_source.shape[0]

    USE_INITIAL_STATE = True
    USE_QK_L2NORM_IN_KERNEL = use_qk_l2norm_in_kernel

    FusedRecurrentGatedDeltaRuleUpdateFwdKernel()(
        q,
        k,
        v,
        g,
        o,
        beta,
        scale,
        initial_state_source,
        initial_state_indices,
        T,
        B,
        H,
        HV,
        K,
        V,
        MAX_T,
        USE_INITIAL_STATE,
        USE_QK_L2NORM_IN_KERNEL,
    )

    return o


def main():
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


if __name__ == "__main__":
    main()

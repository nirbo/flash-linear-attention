# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import warnings

import torch

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
from fla.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o
from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from fla.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd
from fla.ops.utils import chunk_local_cumsum, solve_tril
from fla.utils import CUDAGraphManager, autocast_custom_bwd, autocast_custom_fwd, input_guard


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    output_g: torch.Tensor | None = None,
    output_o: torch.Tensor | None = None,
    output_A: torch.Tensor | None = None,
    output_final_state_buffer: torch.Tensor | None = None,
):
    g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens, output=output_g)
    # obtain WY representation. u is actually the new v.
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        output_dtype=torch.float32,
        output=None, # Use dynamic/temp buffer for KKT to avoid in-place issues in solve_tril
    )
    A = solve_tril(
        A=A,
        cu_seqlens=cu_seqlens,
        output_dtype=k.dtype,
        output=output_A, # Write final result to static buffer
    )
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        output_final_state_buffer=output_final_state_buffer,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        output=output_o,
    )
    return g, o, A, final_state


def chunk_gated_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    output_dq: torch.Tensor | None = None,
    output_dk: torch.Tensor | None = None,
    output_dv: torch.Tensor | None = None,
    output_dg: torch.Tensor | None = None,
    output_dbeta: torch.Tensor | None = None,
):
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
    )
    
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
    )
    dv = chunk_bwd_dv_local(
        q=q,
        k=k,
        g=g,
        do=do,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    if torch.isnan(dv).any():
        print("NaN in dv! Saving debug_inputs.pt")
        torch.save({'q':q, 'k':k, 'g':g, 'do':do, 'scale':scale}, 'debug_inputs.pt')
    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q=q,
        k=k,
        w=w,
        g=g,
        h0=initial_state,
        dht=dht,
        do=do,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        output_dh=None,
        output_dh0=None,
        output_dv2=None, 
    )
    dq, dk, dw, dg = chunk_bwd_dqkwg(
        q=q,
        k=k,
        v=v_new,
        w=w,
        g=g,
        h=h,
        dv=dv,
        do=do,
        dh=dh,
        scale=scale,
        cu_seqlens=cu_seqlens,
        output_dq=output_dq,
        output_dk=output_dk,
        output_dw=None,
        output_dg=None,
    )
    
    # Use explicit output buffers for prepare_wy_repr_bwd if provided
    # Note: prepare_wy_repr_bwd returns dk2, dv, db, dg2
    dk2, dv, db, dg2 = prepare_wy_repr_bwd(
        k=k,
        v=v,
        beta=beta,
        g=g,
        A=A,
        dw=dw,
        du=dv, # passing dv as 'du' argument (gradient w.r.t u/v_new)
        cu_seqlens=cu_seqlens,
        output_dv=output_dv,
        output_db=output_dbeta,
    )
    
    # Accumulate gradients
    if output_dk is not None:
        output_dk.add_(dk2)
        dk = output_dk
    else:
        dk.add_(dk2)
        
    if output_dg is not None:
        # If output_dg was passed to dqkwg, it would be partial.
        # Here we passed None to dqkwg, so dg is a new tensor.
        # We add dg2 to it, then cumsum into output_dg.
        dg.add_(dg2)
        chunk_local_cumsum(dg, chunk_size=64, reverse=True, cu_seqlens=cu_seqlens, output=output_dg)
        dg = output_dg
    else:
        dg.add_(dg2)
        dg = chunk_local_cumsum(dg, chunk_size=64, reverse=True, cu_seqlens=cu_seqlens)

    return dq, dk, dv, db, dg, dh0


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None = None,
        use_qk_l2norm_in_kernel: bool = False,
    ):
        q_rstd, k_rstd = None, None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)

        g, o, A, final_state = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, g, beta, A, initial_state, cu_seqlens)
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dht: torch.Tensor,
    ):
        q, q_rstd, k, k_rstd, v, g, beta, A, initial_state, cu_seqlens = ctx.saved_tensors
        dq, dk, dv, db, dg, dh0 = chunk_gated_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A=A,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            cu_seqlens=cu_seqlens,
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        return dq.to(q), dk.to(k), dv.to(v), dg.to(g), db.to(beta), None, dh0, None, None, None



class ChunkGatedDeltaRuleFunctionGraphSafe(torch.autograd.Function):
    """
    CUDA graph-safe version of ChunkGatedDeltaRuleFunction.

    Uses pre-allocated static buffers from CUDAGraphManager to ensure
    memory addresses remain constant across graph replays.
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor | None,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None,
        use_qk_l2norm_in_kernel: bool,
        cuda_graph_manager: CUDAGraphManager,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if cuda_graph_manager is None:
             raise ValueError("cuda_graph_manager must be provided for GraphSafe function")
             
        # Get static input buffers
        (
            static_q, static_k, static_v, static_g, static_beta, static_initial_state
        ) = cuda_graph_manager.get_input_buffers()

        if use_qk_l2norm_in_kernel:
            # Note: l2norm_fwd allocates new tensors which may be dynamic.
            from fla.modules.l2norm import l2norm_fwd
            static_q, q_rstd = l2norm_fwd(static_q)
            static_k, k_rstd = l2norm_fwd(static_k)

        static_o, static_final_state = cuda_graph_manager.get_output_buffers()
        
        g_out, o_out, A_out, final_state_out = chunk_gated_delta_rule_fwd(
            q=static_q,
            k=static_k,
            v=static_v,
            g=static_g,
            beta=static_beta,
            scale=scale,
            initial_state=static_initial_state if initial_state is not None else None,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            output_g=static_g,
            output_o=static_o,
            output_A=cuda_graph_manager.get_buf_A_view(),
            output_final_state_buffer=static_final_state if output_final_state else None
        )

        # Save static buffers for backward
        # If use_qk_l2norm_in_kernel, q_rstd/k_rstd might be dynamic.
        
        ctx.save_for_backward(
            static_q, q_rstd, static_k, k_rstd, static_v, static_g, static_beta, 
            A_out, static_initial_state, cu_seqlens
        )
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.cuda_graph_manager = cuda_graph_manager # Keep ref to manager
        
        return o_out, final_state_out

    @staticmethod
    @input_guard
    def backward(ctx, do, d_final_state):
        q, q_rstd, k, k_rstd, v, g, beta, A, initial_state, cu_seqlens = ctx.saved_tensors
        manager = ctx.cuda_graph_manager
        
        # Get static gradient buffers
        
        static_dq, static_dk, static_dv, static_dg, static_dbeta = manager.get_grad_buffers()
        
        static_dq, static_dk, static_dv, static_dg, static_dbeta = manager.get_grad_buffers()
        
        chunk_gated_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A=A,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do,
            dht=d_final_state,
            cu_seqlens=cu_seqlens,
            # Outputs
            output_dk=static_dk,
            output_dv=static_dv,
            output_dg=static_dg,
            output_dbeta=static_dbeta
        )
        
        if ctx.use_qk_l2norm_in_kernel:
            # Propagate gradients through l2norm. Note: allocates new dx tensors.
            static_dq = l2norm_bwd(q, q_rstd, static_dq)
            static_dk = l2norm_bwd(k, k_rstd, static_dk)
            
        return static_dq, static_dk, static_dv, static_dg, static_dbeta, None, None, None, None, None, None

def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cuda_graph_manager: CUDAGraphManager | None = None,
    **kwargs
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            (forget) gating tensor (in log space!) of shape `[B, T, H]`.
        beta (torch.Tensor):
            betas of shape `[B, T, H]`.
        scale (Optional[float]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        use_qk_l2norm_in_kernel (bool):
            Whether to apply L2norm to the q/k tensor internally. Default: `False`.
        cuda_graph_manager (Optional[CUDAGraphManager]):
            Manager for static buffers to support CUDA Graph capture. Default: `None`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    if 'head_first' in kwargs:
        warnings.warn(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead.",
        )

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5

    if cuda_graph_manager is not None:
        o, final_state = ChunkGatedDeltaRuleFunctionGraphSafe.apply(
            q,
            k,
            v,
            g,
            beta,
            scale,
            initial_state,
            output_final_state,
            cu_seqlens,
            use_qk_l2norm_in_kernel,
            cuda_graph_manager
        )
    else:
        o, final_state = ChunkGatedDeltaRuleFunction.apply(
            q,
            k,
            v,
            g,
            beta,
            scale,
            initial_state,
            output_final_state,
            cu_seqlens,
            use_qk_l2norm_in_kernel,
        )
    return o, final_state

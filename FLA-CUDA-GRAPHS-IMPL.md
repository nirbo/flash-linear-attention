# FLA CUDA Graphs Implementation Plan

## Document Information

| Field | Value |
|-------|-------|
| Version | 1.0 |
| Status | Draft |
| Target | FLA GatedDeltaNet CUDA Graph Support |
| Approach | Static Memory Pool Allocation |

---

## 1. Executive Summary

This document defines the implementation plan for adding CUDA graph support to FLA's `ChunkGatedDeltaRuleFunction`. The solution uses static memory pool allocation to solve the fundamental incompatibility between PyTorch's autograd saved tensors and CUDA graph memory requirements.

**Scope**: Training-time CUDA graph support for `chunk_gated_delta_rule()` operation.

**Out of Scope**: Other FLA operations (ABC, GLA, HGRN, etc.), inference-only optimizations.

---

## 2. High-Level Design (HLD)

### 2.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Training Loop                                │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌──────────────────┐    ┌────────────────┐ │
│  │  Input Tensors  │───▶│ CUDAGraphManager │───▶│ Static Buffers │ │
│  │  (dynamic addr) │    │                  │    │ (fixed addr)   │ │
│  └─────────────────┘    └──────────────────┘    └───────┬────────┘ │
│                                                         │          │
│  ┌──────────────────────────────────────────────────────▼────────┐ │
│  │              ChunkGatedDeltaRuleFunctionGraphSafe             │ │
│  │  ┌─────────────────┐         ┌─────────────────────────────┐  │ │
│  │  │  Forward Pass   │────────▶│  ctx.save_for_backward()    │  │ │
│  │  │  (Triton kernels)│        │  (saves static buffer refs) │  │ │
│  │  └─────────────────┘         └─────────────────────────────┘  │ │
│  │  ┌─────────────────┐         ┌─────────────────────────────┐  │ │
│  │  │  Backward Pass  │◀────────│  ctx.saved_tensors          │  │ │
│  │  │  (Triton kernels)│        │  (reads static buffer refs) │  │ │
│  │  └─────────────────┘         └─────────────────────────────┘  │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                         │          │
│  ┌──────────────────────────────────────────────────────▼────────┐ │
│  │                      CUDA Graph                               │ │
│  │  - Captures kernel sequence with static addresses             │ │
│  │  - Replays identical operations each iteration                │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Diagram

```
fla/
├── utils.py
│   └── CUDAGraphManager          # NEW: Static buffer manager
│
├── ops/gated_delta_rule/
│   └── chunk.py
│       ├── ChunkGatedDeltaRuleFunction        # EXISTING: Dynamic path
│       ├── ChunkGatedDeltaRuleFunctionGraphSafe  # NEW: Static path
│       └── chunk_gated_delta_rule()           # MODIFY: Add graph_manager param
│
└── layers/
    └── gated_deltanet.py
        └── GatedDeltaNet.forward()            # MODIFY: Add graph_manager param
```

### 2.3 Data Flow

**Without CUDA Graph (current)**:
```
Input → Allocate(dynamic) → Forward → save_for_backward(dynamic_ptr) → Backward → Free
```

**With CUDA Graph (proposed)**:
```
Input → Copy(static_buf) → Forward → save_for_backward(static_ptr) → Backward → [buffers retained]
         ↓
    [Graph Capture]
         ↓
    [Graph Replay] ← same static addresses every iteration
```

---

## 3. Low-Level Design (LLD)

### 3.1 CUDAGraphManager Class

**File**: `fla/utils.py`

**Purpose**: Manages pre-allocated static buffers and CUDA graph lifecycle.

```python
class CUDAGraphManager:
    """
    Manages static memory buffers for CUDA graph-compatible GatedDeltaNet operations.

    The manager pre-allocates all tensors that will be saved for backward pass,
    ensuring their memory addresses remain constant across graph replays.

    Attributes:
        max_batch_size: Maximum supported batch size
        max_seq_len: Maximum supported sequence length
        num_heads: Number of attention heads
        head_dim: Dimension per head (K)
        expand_v: Value expansion factor (V = K * expand_v)
        chunk_size: Chunk size for chunked attention (default: 64)
        device: CUDA device
        dtype: Data type (default: torch.bfloat16)

    Buffers (all pre-allocated):
        q: [B, T, H, K] - Query tensor
        k: [B, T, H, K] - Key tensor
        v: [B, T, H, V] - Value tensor
        g: [B, T, H] - Gating tensor (log-space)
        beta: [B, T, H] - Beta coefficients
        A: [B, NT, H, BT, BT] - WY representation intermediate
        initial_state: [B, H, K, V] - Initial hidden state
        o: [B, T, H, V] - Output tensor
        final_state: [B, H, K, V] - Final hidden state

    Usage:
        manager = CUDAGraphManager(
            max_batch_size=8,
            max_seq_len=2048,
            num_heads=4,
            head_dim=256,
            expand_v=2
        )

        # In training loop:
        o, final_state = chunk_gated_delta_rule(
            q, k, v, g, beta,
            cuda_graph_manager=manager
        )
    """
```

#### 3.1.1 Constructor

```python
def __init__(
    self,
    max_batch_size: int,
    max_seq_len: int,
    num_heads: int,
    head_dim: int,
    expand_v: int = 2,
    chunk_size: int = 64,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
):
    self.max_batch_size = max_batch_size
    self.max_seq_len = max_seq_len
    self.num_heads = num_heads
    self.head_dim = head_dim
    self.expand_v = expand_v
    self.chunk_size = chunk_size
    self.device = device or torch.device('cuda')
    self.dtype = dtype

    # Derived dimensions
    self.value_dim = head_dim * expand_v
    self.num_chunks = (max_seq_len + chunk_size - 1) // chunk_size

    # Allocate buffers
    self._allocate_buffers()

    # Graph state
    self.graph: Optional[torch.cuda.CUDAGraph] = None
    self.warmup_done: bool = False
    self.captured: bool = False
```

#### 3.1.2 Buffer Allocation

```python
def _allocate_buffers(self) -> None:
    """Pre-allocate all static buffers."""
    B = self.max_batch_size
    T = self.max_seq_len
    H = self.num_heads
    K = self.head_dim
    V = self.value_dim
    NT = self.num_chunks
    BT = self.chunk_size

    # Input buffers (copied from dynamic inputs)
    self.buf_q = torch.empty(B, T, H, K, device=self.device, dtype=self.dtype)
    self.buf_k = torch.empty(B, T, H, K, device=self.device, dtype=self.dtype)
    self.buf_v = torch.empty(B, T, H, V, device=self.device, dtype=self.dtype)
    self.buf_g = torch.empty(B, T, H, device=self.device, dtype=self.dtype)
    self.buf_beta = torch.empty(B, T, H, device=self.device, dtype=self.dtype)

    # Optional input buffers
    self.buf_initial_state = torch.empty(B, H, K, V, device=self.device, dtype=self.dtype)

    # Intermediate buffers (computed during forward)
    self.buf_A = torch.empty(B, NT, H, BT, BT, device=self.device, dtype=torch.float32)

    # Optional normalization stats
    self.buf_q_rstd = torch.empty(B, T, H, device=self.device, dtype=torch.float32)
    self.buf_k_rstd = torch.empty(B, T, H, device=self.device, dtype=torch.float32)

    # Output buffers
    self.buf_o = torch.empty(B, T, H, V, device=self.device, dtype=self.dtype)
    self.buf_final_state = torch.empty(B, H, K, V, device=self.device, dtype=self.dtype)

    # Gradient buffers (for backward pass)
    self.buf_dq = torch.empty(B, T, H, K, device=self.device, dtype=self.dtype)
    self.buf_dk = torch.empty(B, T, H, K, device=self.device, dtype=self.dtype)
    self.buf_dv = torch.empty(B, T, H, V, device=self.device, dtype=self.dtype)
    self.buf_dg = torch.empty(B, T, H, device=self.device, dtype=self.dtype)
    self.buf_dbeta = torch.empty(B, T, H, device=self.device, dtype=self.dtype)

    # Track actual dimensions for current batch
    self._current_B = 0
    self._current_T = 0
```

#### 3.1.3 Buffer Copy Methods

```python
def copy_inputs(
    self,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    initial_state: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    """
    Copy dynamic input tensors to static buffers.

    Returns references to static buffers (sliced to actual size).
    """
    B, T, H, K = q.shape
    V = v.shape[-1]

    # Validate dimensions
    if B > self.max_batch_size:
        raise ValueError(f"Batch size {B} exceeds max {self.max_batch_size}")
    if T > self.max_seq_len:
        raise ValueError(f"Seq len {T} exceeds max {self.max_seq_len}")

    # Track current dimensions
    self._current_B = B
    self._current_T = T

    # Copy to static buffers (in-place)
    self.buf_q[:B, :T].copy_(q)
    self.buf_k[:B, :T].copy_(k)
    self.buf_v[:B, :T].copy_(v)
    self.buf_g[:B, :T].copy_(g)
    self.buf_beta[:B, :T].copy_(beta)

    if initial_state is not None:
        self.buf_initial_state[:B].copy_(initial_state)
        static_initial_state = self.buf_initial_state[:B]
    else:
        static_initial_state = None

    # Return sliced views (same memory, correct shape)
    return (
        self.buf_q[:B, :T],
        self.buf_k[:B, :T],
        self.buf_v[:B, :T],
        self.buf_g[:B, :T],
        self.buf_beta[:B, :T],
        static_initial_state,
    )

def get_output_buffers(self) -> Tuple[Tensor, Tensor]:
    """Get sliced output buffers for current batch dimensions."""
    B, T = self._current_B, self._current_T
    return self.buf_o[:B, :T], self.buf_final_state[:B]

def get_grad_buffers(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Get sliced gradient buffers for current batch dimensions."""
    B, T = self._current_B, self._current_T
    return (
        self.buf_dq[:B, :T],
        self.buf_dk[:B, :T],
        self.buf_dv[:B, :T],
        self.buf_dg[:B, :T],
        self.buf_dbeta[:B, :T],
    )
```

#### 3.1.4 Memory Reporting

```python
def memory_footprint(self) -> int:
    """Calculate total memory footprint in bytes."""
    total = 0
    for name in dir(self):
        if name.startswith('buf_'):
            buf = getattr(self, name)
            if isinstance(buf, torch.Tensor):
                total += buf.numel() * buf.element_size()
    return total

def __repr__(self) -> str:
    mem_mb = self.memory_footprint() / (1024 * 1024)
    return (
        f"CUDAGraphManager("
        f"max_batch={self.max_batch_size}, "
        f"max_seq={self.max_seq_len}, "
        f"heads={self.num_heads}, "
        f"head_dim={self.head_dim}, "
        f"memory={mem_mb:.1f}MB)"
    )
```

---

### 3.2 ChunkGatedDeltaRuleFunctionGraphSafe

**File**: `fla/ops/gated_delta_rule/chunk.py`

**Purpose**: Graph-safe autograd.Function that uses static buffers.

#### 3.2.1 Class Definition

```python
class ChunkGatedDeltaRuleFunctionGraphSafe(torch.autograd.Function):
    """
    CUDA graph-safe version of ChunkGatedDeltaRuleFunction.

    Uses pre-allocated static buffers from CUDAGraphManager to ensure
    memory addresses remain constant across graph replays.

    Key differences from ChunkGatedDeltaRuleFunction:
    1. Accepts cuda_graph_manager parameter
    2. Copies inputs to static buffers before forward
    3. Saves static buffer references (not dynamic tensors)
    4. Uses static gradient buffers in backward
    """
```

#### 3.2.2 Forward Pass

```python
@staticmethod
def forward(
    ctx,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    scale: float,
    initial_state: Optional[Tensor],
    output_final_state: bool,
    cu_seqlens: Optional[Tensor],
    use_qk_l2norm_in_kernel: bool,
    cuda_graph_manager: CUDAGraphManager,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Forward pass using static buffers.

    Steps:
    1. Copy inputs to static buffers
    2. Run forward kernels on static buffers
    3. Save static buffer references for backward
    4. Copy output to dynamic tensor for return
    """
    # Step 1: Copy to static buffers
    (
        static_q, static_k, static_v, static_g, static_beta, static_initial_state
    ) = cuda_graph_manager.copy_inputs(q, k, v, g, beta, initial_state)

    # Step 2: Run forward (operates on static buffers)
    # Note: Output is written directly to static output buffer
    g_cumsum, o, A, final_state, q_rstd, k_rstd = chunk_gated_delta_rule_fwd(
        q=static_q,
        k=static_k,
        v=static_v,
        g=static_g,
        beta=static_beta,
        scale=scale,
        initial_state=static_initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )

    # Step 3: Save static buffer references
    # These addresses will not change between graph replays
    ctx.save_for_backward(
        static_q,      # Static buffer reference
        q_rstd,        # May be None
        static_k,      # Static buffer reference
        k_rstd,        # May be None
        static_v,      # Static buffer reference
        static_g,      # Static buffer reference
        static_beta,   # Static buffer reference
        A,             # Intermediate (allocated in forward)
        static_initial_state,  # Static buffer reference (may be None)
        cu_seqlens,    # Usually None for non-varlen
    )

    # Save non-tensor context
    ctx.scale = scale
    ctx.output_final_state = output_final_state
    ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
    ctx.cuda_graph_manager = cuda_graph_manager

    # Step 4: Copy output to dynamic tensor
    # This allows the output to participate in downstream dynamic graph
    o_out = o.clone()
    final_state_out = final_state.clone() if output_final_state else None

    return o_out, final_state_out
```

#### 3.2.3 Backward Pass

```python
@staticmethod
def backward(
    ctx,
    do: Tensor,
    d_final_state: Optional[Tensor],
) -> Tuple[Optional[Tensor], ...]:
    """
    Backward pass using static buffers.

    Steps:
    1. Retrieve static buffer references from ctx
    2. Copy gradient input to static buffer
    3. Run backward kernels on static buffers
    4. Copy gradients to dynamic tensors for return
    """
    # Step 1: Retrieve saved static references
    (
        q, q_rstd, k, k_rstd, v, g, beta, A, initial_state, cu_seqlens
    ) = ctx.saved_tensors

    manager = ctx.cuda_graph_manager

    # Step 2: Copy gradient input to static buffer (if needed)
    # Note: do comes from upstream, may be dynamic
    B, T, H, V = do.shape
    static_do = manager.buf_o[:B, :T]  # Reuse output buffer for grad
    static_do.copy_(do)

    # Step 3: Run backward (operates on static buffers)
    dq, dk, dv, dg, dbeta, d_initial_state = chunk_gated_delta_rule_bwd(
        q=q,
        q_rstd=q_rstd,
        k=k,
        k_rstd=k_rstd,
        v=v,
        g=g,
        beta=beta,
        A=A,
        initial_state=initial_state,
        do=static_do,
        dht=d_final_state,
        scale=ctx.scale,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=ctx.use_qk_l2norm_in_kernel,
    )

    # Step 4: Copy gradients to dynamic tensors
    # This allows gradients to flow to upstream dynamic graph
    dq_out = dq.clone()
    dk_out = dk.clone()
    dv_out = dv.clone()
    dg_out = dg.clone()
    dbeta_out = dbeta.clone()
    d_initial_state_out = d_initial_state.clone() if d_initial_state is not None else None

    return (
        dq_out,              # dq
        dk_out,              # dk
        dv_out,              # dv
        dg_out,              # dg
        dbeta_out,           # dbeta
        None,                # scale (not differentiable)
        d_initial_state_out, # d_initial_state
        None,                # output_final_state
        None,                # cu_seqlens
        None,                # use_qk_l2norm_in_kernel
        None,                # cuda_graph_manager
    )
```

---

### 3.3 Entry Point Modification

**File**: `fla/ops/gated_delta_rule/chunk.py`

**Function**: `chunk_gated_delta_rule()`

#### 3.3.1 Signature Update

```python
def chunk_gated_delta_rule(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[Tensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cuda_graph_manager: Optional[CUDAGraphManager] = None,  # NEW PARAMETER
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Compute chunk-wise gated delta rule attention.

    Args:
        ... (existing args) ...
        cuda_graph_manager: Optional CUDAGraphManager for CUDA graph support.
            When provided, uses static buffer allocation for graph compatibility.
            When None, uses standard dynamic allocation (original behavior).

    Returns:
        o: Output tensor [B, T, H, V]
        final_state: Final hidden state [B, H, K, V] if output_final_state else None
    """
```

#### 3.3.2 Implementation Update

```python
# Remove @torch.compiler.disable for graph-safe path
def chunk_gated_delta_rule(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[Tensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cuda_graph_manager: Optional[CUDAGraphManager] = None,
) -> Tuple[Tensor, Optional[Tensor]]:

    # Existing preprocessing (head_first transpose, etc.)
    if head_first:
        q, k, v, g, beta = map(lambda x: rearrange(x, 'b h t d -> b t h d'), (q, k, v, g, beta))

    if scale is None:
        scale = q.shape[-1] ** -0.5

    # Route to appropriate implementation
    if cuda_graph_manager is not None:
        # Graph-safe path with static buffers
        o, final_state = ChunkGatedDeltaRuleFunctionGraphSafe.apply(
            q, k, v, g, beta,
            scale,
            initial_state,
            output_final_state,
            cu_seqlens,
            use_qk_l2norm_in_kernel,
            cuda_graph_manager,
        )
    else:
        # Original dynamic path (unchanged)
        o, final_state = ChunkGatedDeltaRuleFunction.apply(
            q, k, v, g, beta,
            scale,
            initial_state,
            output_final_state,
            cu_seqlens,
            use_qk_l2norm_in_kernel,
        )

    if head_first:
        o = rearrange(o, 'b t h d -> b h t d')

    return o, final_state
```

---

### 3.4 Layer Integration

**File**: `fla/layers/gated_deltanet.py`

**Class**: `GatedDeltaNet`

#### 3.4.1 Forward Signature Update

```python
def forward(
    self,
    hidden_states: Tensor,
    attention_mask: Optional[Tensor] = None,
    past_key_values: Optional[Tensor] = None,
    use_cache: bool = False,
    output_attentions: bool = False,
    cuda_graph_manager: Optional[CUDAGraphManager] = None,  # NEW PARAMETER
) -> Tuple[Tensor, ...]:
    """
    Forward pass for GatedDeltaNet layer.

    Args:
        ... (existing args) ...
        cuda_graph_manager: Optional CUDAGraphManager for CUDA graph support.
    """
```

#### 3.4.2 Forward Implementation Update

```python
def forward(
    self,
    hidden_states: Tensor,
    attention_mask: Optional[Tensor] = None,
    past_key_values: Optional[Tensor] = None,
    use_cache: bool = False,
    output_attentions: bool = False,
    cuda_graph_manager: Optional[CUDAGraphManager] = None,
) -> Tuple[Tensor, ...]:

    # ... existing projection code ...
    q = self.q_proj(hidden_states)
    k = self.k_proj(hidden_states)
    v = self.v_proj(hidden_states)
    g = self.g_proj(hidden_states)
    beta = self.beta_proj(hidden_states)

    # ... existing reshape code ...

    # Call kernel with graph manager
    o, final_state = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=self.scale,
        initial_state=past_key_values,
        output_final_state=use_cache,
        cuda_graph_manager=cuda_graph_manager,  # Pass through
    )

    # ... existing output projection code ...
```

---

## 4. Integration Guide

### 4.1 Training Loop Integration

```python
from fla.utils import CUDAGraphManager

# Configuration
MAX_BATCH_SIZE = 8
MAX_SEQ_LEN = 2048
NUM_HEADS = 4
HEAD_DIM = 256
EXPAND_V = 2

# Create manager (once at start)
graph_manager = CUDAGraphManager(
    max_batch_size=MAX_BATCH_SIZE,
    max_seq_len=MAX_SEQ_LEN,
    num_heads=NUM_HEADS,
    head_dim=HEAD_DIM,
    expand_v=EXPAND_V,
    device=torch.device('cuda'),
    dtype=torch.bfloat16,
)

print(f"Graph manager allocated: {graph_manager}")
# Output: CUDAGraphManager(max_batch=8, max_seq=2048, heads=4, head_dim=256, memory=XXX.XMB)

# Model setup
model = GatedDeltaNetModel(...)
optimizer = torch.optim.AdamW(model.parameters())

# Training loop
for step, batch in enumerate(dataloader):
    input_ids = batch['input_ids'].cuda()

    # Forward pass with graph manager
    outputs = model(
        input_ids,
        cuda_graph_manager=graph_manager,
    )
    loss = outputs.loss

    # Backward
    loss.backward()

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()
```

### 4.2 Multi-Layer Model Integration

```python
class GatedDeltaNetModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([
            GatedDeltaNet(config) for _ in range(config.num_layers)
        ])

    def forward(
        self,
        hidden_states: Tensor,
        cuda_graph_manager: Optional[CUDAGraphManager] = None,
    ) -> Tensor:
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                cuda_graph_manager=cuda_graph_manager,
            )[0]
        return hidden_states
```

### 4.3 Per-Layer Graph Managers (Alternative)

For models with different layer configurations:

```python
# Create per-layer managers if layers have different dimensions
graph_managers = []
for layer_config in layer_configs:
    manager = CUDAGraphManager(
        max_batch_size=MAX_BATCH_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        num_heads=layer_config.num_heads,
        head_dim=layer_config.head_dim,
        expand_v=layer_config.expand_v,
    )
    graph_managers.append(manager)

# In forward
for layer, manager in zip(model.layers, graph_managers):
    hidden_states = layer(hidden_states, cuda_graph_manager=manager)[0]
```

---

## 5. Test Specifications

### 5.1 Unit Tests

**File**: `tests/test_cuda_graph_support.py`

#### Test 5.1.1: Buffer Allocation

```python
def test_buffer_allocation():
    """Verify buffer dimensions and memory footprint."""
    manager = CUDAGraphManager(
        max_batch_size=4,
        max_seq_len=1024,
        num_heads=4,
        head_dim=128,
        expand_v=2,
    )

    # Check buffer shapes
    assert manager.buf_q.shape == (4, 1024, 4, 128)
    assert manager.buf_v.shape == (4, 1024, 4, 256)
    assert manager.buf_o.shape == (4, 1024, 4, 256)

    # Check memory footprint is reasonable
    mem_mb = manager.memory_footprint() / (1024 * 1024)
    assert 50 < mem_mb < 200  # Expected range
```

#### Test 5.1.2: Input Copy

```python
def test_input_copy():
    """Verify inputs are correctly copied to static buffers."""
    manager = CUDAGraphManager(...)

    # Create dynamic inputs
    q = torch.randn(2, 512, 4, 128, device='cuda')
    k = torch.randn(2, 512, 4, 128, device='cuda')
    v = torch.randn(2, 512, 4, 256, device='cuda')
    g = torch.randn(2, 512, 4, device='cuda')
    beta = torch.randn(2, 512, 4, device='cuda')

    # Copy to static buffers
    static_q, static_k, static_v, static_g, static_beta, _ = manager.copy_inputs(
        q, k, v, g, beta
    )

    # Verify data matches
    torch.testing.assert_close(static_q, q)
    torch.testing.assert_close(static_k, k)
    torch.testing.assert_close(static_v, v)

    # Verify static buffers are views of manager buffers
    assert static_q.data_ptr() == manager.buf_q[:2, :512].data_ptr()
```

#### Test 5.1.3: Dimension Validation

```python
def test_dimension_validation():
    """Verify dimension limits are enforced."""
    manager = CUDAGraphManager(
        max_batch_size=4,
        max_seq_len=1024,
        ...
    )

    # Should raise for exceeding batch size
    with pytest.raises(ValueError, match="Batch size.*exceeds max"):
        q = torch.randn(8, 512, 4, 128, device='cuda')  # B=8 > max=4
        manager.copy_inputs(q, ...)

    # Should raise for exceeding seq len
    with pytest.raises(ValueError, match="Seq len.*exceeds max"):
        q = torch.randn(2, 2048, 4, 128, device='cuda')  # T=2048 > max=1024
        manager.copy_inputs(q, ...)
```

### 5.2 Gradient Correctness Tests

#### Test 5.2.1: Basic Gradient Match

```python
def test_gradient_correctness():
    """Verify gradients match between dynamic and static paths."""
    torch.manual_seed(42)
    B, T, H, K, V = 2, 256, 4, 64, 128

    # Inputs (shared)
    q = torch.randn(B, T, H, K, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, T, H, K, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, T, H, V, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    g = torch.randn(B, T, H, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    beta = torch.randn(B, T, H, device='cuda', dtype=torch.bfloat16, requires_grad=True)

    # Reference: dynamic path
    o_ref, _ = chunk_gated_delta_rule(q, k, v, g, beta, cuda_graph_manager=None)
    loss_ref = o_ref.sum()
    loss_ref.backward()
    grads_ref = {name: p.grad.clone() for name, p in [('q', q), ('k', k), ('v', v), ('g', g), ('beta', beta)]}

    # Zero grads
    for p in [q, k, v, g, beta]:
        p.grad = None

    # Test: static path
    manager = CUDAGraphManager(max_batch_size=B, max_seq_len=T, num_heads=H, head_dim=K, expand_v=V//K)
    o_test, _ = chunk_gated_delta_rule(q, k, v, g, beta, cuda_graph_manager=manager)
    loss_test = o_test.sum()
    loss_test.backward()
    grads_test = {name: p.grad for name, p in [('q', q), ('k', k), ('v', v), ('g', g), ('beta', beta)]}

    # Compare
    for name in grads_ref:
        torch.testing.assert_close(
            grads_ref[name], grads_test[name],
            rtol=1e-3, atol=1e-3,
            msg=f"Gradient mismatch for {name}"
        )
```

#### Test 5.2.2: Parametrized Gradient Tests

```python
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize("seq_len", [128, 256, 512, 1024])
@pytest.mark.parametrize("with_initial_state", [False, True])
def test_gradient_correctness_parametrized(batch_size, seq_len, with_initial_state):
    """Test gradient correctness across configurations."""
    # ... similar to test_gradient_correctness ...
```

### 5.3 Performance Benchmarks

#### Benchmark 5.3.1: Forward Latency

```python
def benchmark_forward_latency():
    """Measure forward pass latency improvement."""
    configs = [
        {'B': 1, 'T': 512},   # Small batch (should see big improvement)
        {'B': 4, 'T': 1024},  # Medium batch
        {'B': 8, 'T': 2048},  # Large batch (less improvement)
    ]

    for config in configs:
        B, T = config['B'], config['T']
        q = torch.randn(B, T, 4, 256, device='cuda', dtype=torch.bfloat16)
        # ... setup ...

        # Warmup
        for _ in range(10):
            o, _ = chunk_gated_delta_rule(q, k, v, g, beta)

        # Benchmark dynamic
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            o, _ = chunk_gated_delta_rule(q, k, v, g, beta, cuda_graph_manager=None)
        torch.cuda.synchronize()
        time_dynamic = (time.perf_counter() - start) / 100

        # Benchmark static
        manager = CUDAGraphManager(...)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            o, _ = chunk_gated_delta_rule(q, k, v, g, beta, cuda_graph_manager=manager)
        torch.cuda.synchronize()
        time_static = (time.perf_counter() - start) / 100

        speedup = time_dynamic / time_static
        print(f"B={B}, T={T}: dynamic={time_dynamic*1000:.2f}ms, static={time_static*1000:.2f}ms, speedup={speedup:.2f}x")
```

---

## 6. Implementation Schedule

### Phase 1: Core Implementation (Weeks 1-4)

| Week | Tasks | Deliverables |
|------|-------|--------------|
| 1 | Implement `CUDAGraphManager` class | `fla/utils.py` updated |
| 2 | Implement `ChunkGatedDeltaRuleFunctionGraphSafe` | `chunk.py` updated |
| 3 | Update entry point and layer integration | End-to-end path working |
| 4 | Basic unit tests | `tests/test_cuda_graph_support.py` |

### Phase 2: Validation (Weeks 5-6)

| Week | Tasks | Deliverables |
|------|-------|--------------|
| 5 | Gradient correctness tests | All grad tests passing |
| 6 | Performance benchmarks | Benchmark results documented |

### Phase 3: Polish (Weeks 7-8)

| Week | Tasks | Deliverables |
|------|-------|--------------|
| 7 | Edge cases, error handling | Robust implementation |
| 8 | Documentation, examples | Usage guide, examples |

---

## 7. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Gradient mismatch | Medium | High | Extensive gradient tests at each config |
| Memory fragmentation | Low | Medium | Use contiguous allocation, monitor fragmentation |
| Triton kernel incompatibility | Low | High | Test with all kernel variants |
| Variable seq length issues | Medium | Medium | Implement padding strategy, test thoroughly |
| Clone overhead negates gains | Medium | Medium | Profile clone vs kernel launch, optimize if needed |

---

## 8. Success Criteria

1. **Correctness**: All gradient tests pass with rtol=1e-3, atol=1e-3
2. **Performance**: >10% throughput improvement for batch_size <= 4
3. **Memory**: <20% memory overhead from static buffers
4. **Stability**: No crashes or NaN gradients in 10,000 step training run
5. **Compatibility**: Works with torch.compile(mode="default")

---

## 9. Open Questions

1. **Buffer sharing across layers**: Should all layers share one manager, or each layer have its own?
   - Recommendation: Start with per-model manager, optimize later if needed

2. **Variable sequence length strategy**: Pad to max, or multiple managers for common lengths?
   - Recommendation: Pad to max initially, add bucketing optimization later

3. **Clone overhead**: Is the copy from static→dynamic too expensive?
   - Answer: Profile first, likely negligible compared to kernel execution

4. **Upstream acceptance**: Will FLA maintainers accept this approach?
   - Mitigation: Start with RFC/issue to gauge interest before full implementation

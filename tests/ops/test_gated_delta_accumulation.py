
import torch
import torch.nn as nn
from fla.layers.gated_deltanet import GatedDeltaNet
from fla.utils import CUDAGraphManager

def test_grad_accumulation_manager():
    print("Testing Gradient Accumulation with CUDAGraphManager...")
    
    B, T, H, D = 4, 128, 4, 256
    V = 512
    device = 'cuda'
    dtype = torch.bfloat16
    
    # 1. Setup Layer and Manager
    model = GatedDeltaNet(
        hidden_size=D, 
        expand_v=2.0, 
        head_dim=D//H, 
        num_heads=H
    ).to(device=device, dtype=dtype)
    
    manager = CUDAGraphManager(
        max_batch_size=B,
        max_seq_len=T,
        num_heads=H,
        head_dim=D//H,
        expand_v=2, # Matches model expand_v
        dtype=dtype,
        device=device
    )
    
    # Enable compile to ensure we hit the GraphSafe path logic
    # (though manager usage forces it regardless of compile)
    # model = torch.compile(model, mode='max-autotune') 
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # 2. Gradient Accumulation Loop
    # We simulate 2 accumulation steps before optimizer.step()
    
    model.train()
    optimizer.zero_grad()
    
    print("Accumulation Step 1")
    x1 = torch.randn(B, T, D, device=device, dtype=dtype)
    # Forward Pass 1
    # Note: GatedDeltaNet computes q,k,v internally. 
    # Manager copies them to static buffers inside forward().
    out1, _, _ = model(x1, cuda_graph_manager=manager)
    loss1 = out1.sum()
    
    # Retain graph effectively keeps the Backward Graph alive? 
    # Or just calling backward() keeps leaf tensors alive?
    # Standard grad accum calls backward separately.
    loss1.backward() 
    
    # At this point, static buffers contain Batch 1 data.
    # Saved tensors for Backward 1 have been consumed (unless create_graph=True).
    # But if we had `create_graph=True` (like Sophia), it would crash if buffers mutated?
    
    print("Accumulation Step 2 (This creates the conflict if buffers reused wrongly)")
    x2 = torch.randn(B, T, D, device=device, dtype=dtype)
    
    # Forward Pass 2
    # This calls manager.copy_inputs(x2_q, ...), overwriting static buffers!
    # If Backward 1 needed static buffers and they were somehow kept alive, we'd crash.
    # BUT standard backward() frees the graph.
    # The user reported crash with "grad accumulation > 1".
    # And specifically "modified by an inplace operation".
    # This usually happens if the graph is trying to check version counters.
    
    out2, _, _ = model(x2, cuda_graph_manager=manager)
    loss2 = out2.sum()
    loss2.backward()
    
    optimizer.step()
    print("Optimizer Step Successful")
    
    # 3. Test with create_graph=True (Sophia-style)
    print("Testing with create_graph=True (Sophia style)")
    optimizer.zero_grad()
    
    x3 = torch.randn(B, T, D, device=device, dtype=dtype)
    out3, _, _ = model(x3, cuda_graph_manager=manager)
    loss3 = out3.sum()
    loss3.backward(create_graph=True)
    
    x4 = torch.randn(B, T, D, device=device, dtype=dtype)
    out4, _, _ = model(x4, cuda_graph_manager=manager)
    loss4 = out4.sum()
    try:
        loss4.backward(create_graph=True)
        print("Success: create_graph=True accumulation passed")
    except RuntimeError as e:
        print(f"Caught Expected Error or Unexpected?: {e}")
        # If we fixed it, this should NOT error.
        raise e

if __name__ == "__main__":
    test_grad_accumulation_manager()

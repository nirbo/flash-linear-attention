
import torch
import torch.nn as nn

from fla.layers.gated_deltanet import GatedDeltaNet

# Set env var for torch compile debugging potentially
# os.environ["TORCH_LOGS"] = "+inductor"

class SimpleGDNModel(nn.Module):
    def __init__(self, d_model, n_heads, n_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            GatedDeltaNet(
                hidden_size=d_model,
                num_heads=n_heads,
                use_gate=True,
                use_short_conv=True,
                conv_size=4,
                share_conv_kernel=True
            ) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 100) # Toy output

    def forward(self, x):
        # x: [B, T, D]
        for layer in self.layers:
            x, _, _ = layer(x)
        x = self.norm(x)
        logits = self.head(x)
        return logits

def train_loop(model, optimizer, B, T, D, steps=20):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    
    # Static inputs for graph simulation style
    for i in range(steps):
        x = torch.randn(B, T, D, device='cuda', dtype=torch.bfloat16)
        y = torch.randint(0, 100, (B, T), device='cuda')
        
        optimizer.zero_grad()
        logits = model(x)
        # logits: [B, T, 100]
        loss = loss_fn(logits.view(-1, 100), y.view(-1))
        
        loss.backward()
        
        # Clip grad for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        if i % 5 == 0:
            print(f"Step {i}: Loss {loss.item()}")
            
def test_compile_max_autotune():
    print("Testing torch.compile(mode='max-autotune') with GDN layers...")
    B, T, D, H = 4, 128, 256, 4
    model = SimpleGDNModel(d_model=D, n_heads=H).cuda().to(torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Compile
    # Note: 'max-autotune' enables CUDA Graphs in Inductor.
    compiled_model = torch.compile(model, mode='max-autotune')
    
    print("Starting Training Loop...")
    try:
        train_loop(compiled_model, optimizer, B, T, D, steps=20)
        print("Training Loop Completed Successfully!")
    except Exception as e:
        print(f"FAILED with error: {e}")
        raise e

if __name__ == "__main__":
    torch.manual_seed(42)
    test_compile_max_autotune()

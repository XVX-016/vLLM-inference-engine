"""
Per-step attention output diagnostic.
Compares HF attention output (from a hooked version) vs our kernel for the SAME
Q, K_cache, V_cache at each decode step. Finds the first step where they disagree.
"""
import sys, torch, math
sys.path.insert(0, r'c:\Computing\vLLM-inference-engine')
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from engine.kernels.paged_attention import paged_attention_decode_v2
from engine.memory.block_allocator import BlockAllocator, BlockTable, PhysicalBlockPool

PROMPT = "The capital of France is"
MAX_TOKENS = 18  # just past where divergence was reported (step 13)
BLOCK_SIZE = 16
device = "cuda"
TARGET_LAYER = 0  # Check layer 0 first

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float16, device_map=device
)
model.eval()

cfg = model.config
num_layers = cfg.num_hidden_layers
num_q_heads = cfg.num_attention_heads      # 14
num_kv_heads = cfg.num_key_value_heads     # 2
head_dim = cfg.hidden_size // num_q_heads  # 64
scale = 1.0 / math.sqrt(head_dim)
group_size = num_q_heads // num_kv_heads   # 7

enc = tokenizer(PROMPT, return_tensors="pt").to(device)
input_ids = enc.input_ids
prompt_len = input_ids.shape[1]
print(f"prompt_len={prompt_len}, num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}")

# ----- Set up paged pool -----
pool = PhysicalBlockPool(
    num_blocks=512, block_size=BLOCK_SIZE, num_layers=num_layers,
    num_kv_heads=num_kv_heads, head_dim=head_dim,
    dtype=torch.float16, device=device,
)
alloc = BlockAllocator(num_blocks=512)
bt = BlockTable(seq_id=0, block_size=BLOCK_SIZE)

# ----- Prefill -----
with torch.inference_mode():
    prefill_out = model(input_ids=input_ids, use_cache=True)
    pkv = prefill_out.past_key_values

# Copy prefill K/V into pool
for t in range(prompt_len):
    if not bt.last_block_has_space():
        bt.physical_block_ids.append(alloc.allocate())
    bt.num_tokens += 1
    lb = t // BLOCK_SIZE
    pb = bt.physical_block_ids[lb]
    off = t % BLOCK_SIZE
    for li in range(num_layers):
        pool.k_cache[li][pb, off] = pkv.layers[li].keys[0, :, t, :]
        pool.v_cache[li][pb, off] = pkv.layers[li].values[0, :, t, :]

# ----- Hook to capture Q after RoPE in a specific layer -----
captured = {}

def make_capture_hook(layer_idx):
    """Captures Q, K (new token), V (new token) at the point of writing to pool."""
    def hook_fn(module, args, kwargs, output):
        # After RoPE, inside our patched forward
        # We'll capture Q by hooking into q_proj then RoPE
        pass
    return hook_fn

# Actually, let's hook q_proj output and manually compute RoPE
attn_module = model.model.layers[TARGET_LAYER].self_attn

print("\nStep-by-step comparison (HF SDPA vs our kernel, layer 0 only):")
print(f"{'Step':>4} {'SeqLen':>7} {'MaxDiff':>10}  Status")
print("-" * 40)

next_token = torch.argmax(prefill_out.logits[:, -1, :], dim=-1, keepdim=True)
seq_len = prompt_len

with torch.inference_mode():
    for step in range(1, MAX_TOKENS):
        seq_len += 1
        if not bt.last_block_has_space():
            bt.physical_block_ids.append(alloc.allocate())
        bt.num_tokens += 1

        pos_id = torch.tensor([[seq_len - 1]], dtype=torch.long, device=device)

        # --- Capture Q and new K/V from this decode step, layer 0 ---
        # Get hidden states up to layer 0 by running embeddings only
        hidden = model.model.embed_tokens(next_token)  # [1, 1, hidden_size]
        # Run RMS norms and layer 0 self_attn projections manually
        normed = model.model.layers[TARGET_LAYER].input_layernorm(hidden)  # [1, 1, hidden_size]
        
        # Projections
        q = attn_module.q_proj(normed)  # [1, 1, num_q_heads*head_dim]
        k_new = attn_module.k_proj(normed)
        v_new = attn_module.v_proj(normed)
        
        # Reshape to [1, num_heads, 1, head_dim]
        q = q.view(1, 1, num_q_heads, head_dim).transpose(1, 2)
        k_new = k_new.view(1, 1, num_kv_heads, head_dim).transpose(1, 2)
        v_new = v_new.view(1, 1, num_kv_heads, head_dim).transpose(1, 2)
        
        # RoPE
        cos, sin = attn_module.rotary_emb(q, position_ids=pos_id)
        q_rope, k_rope = apply_rotary_pos_emb(q, k_new, cos, sin)
        
        # Write new K/V to pool
        token_pos = seq_len - 1
        lb = token_pos // BLOCK_SIZE
        pb = bt.physical_block_ids[lb]
        off = token_pos % BLOCK_SIZE
        pool.k_cache[TARGET_LAYER][pb, off] = k_rope.squeeze(0).squeeze(1)
        pool.v_cache[TARGET_LAYER][pb, off] = v_new.squeeze(0).squeeze(1)
        
        # --- SDPA reference: gather K/V from pool ---
        num_blocks_for_seq = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        k_full = torch.zeros(1, num_kv_heads, seq_len, head_dim, dtype=torch.float16, device=device)
        v_full = torch.zeros(1, num_kv_heads, seq_len, head_dim, dtype=torch.float16, device=device)
        for t in range(seq_len):
            b_idx = t // BLOCK_SIZE
            phys = bt.physical_block_ids[b_idx]
            off_t = t % BLOCK_SIZE
            k_full[0, :, t, :] = pool.k_cache[TARGET_LAYER][phys, off_t]
            v_full[0, :, t, :] = pool.v_cache[TARGET_LAYER][phys, off_t]
        
        # Expand for GQA
        k_exp = k_full.repeat_interleave(group_size, dim=1)
        v_exp = v_full.repeat_interleave(group_size, dim=1)
        sdpa_out = torch.nn.functional.scaled_dot_product_attention(
            q_rope, k_exp, v_exp, scale=scale
        ).squeeze(0).squeeze(1)  # [num_q_heads, head_dim]
        
        # --- Kernel output ---
        block_table_tensor = torch.tensor(bt.physical_block_ids[:num_blocks_for_seq],
                                           dtype=torch.int32, device=device)
        kernel_out = paged_attention_decode_v2(
            q=q_rope.squeeze(0).squeeze(1),  # [num_q_heads, head_dim]
            k_cache=pool.k_cache[TARGET_LAYER],
            v_cache=pool.v_cache[TARGET_LAYER],
            block_table=block_table_tensor,
            seq_len=seq_len,
            scale=scale,
        )
        
        diff = (kernel_out.float() - sdpa_out.float()).abs()
        max_d = diff.max().item()
        status = "OK" if max_d < 0.1 else f"DIVERGED! max_diff={max_d:.4f}"
        print(f"{step:>4} {seq_len:>7} {max_d:>10.4f}  {status}")
        
        # Advance token
        next_token = next_token  # keep same token for simplicity (we're only checking attention)

print("\nDone.")

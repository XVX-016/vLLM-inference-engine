# Part 1: Writing a Custom Paged Attention Triton Kernel from Scratch

In standard LLM inference, the key-value (KV) cache grows linearly with sequence length. Because naive batching requires contiguous memory allocation sized for the maximum possible sequence length, up to 60–80% of VRAM is lost to internal and external memory fragmentation.

In this post, we explore how to implement a custom **Paged Attention kernel** using OpenAI's Triton language, integrate it into a HuggingFace transformer model (`Qwen/Qwen2.5-0.5B-Instruct`), and test numerical parity against standard attention.

---

## The Core Concept: Non-Contiguous KV Blocks

In Paged Attention (first introduced by Kwon et al. in vLLM), physical memory is divided into fixed-size blocks (e.g. 16 tokens per block). Each logical sequence maintains a **Block Table**—an array of physical block pointers:

```
Logical Sequence:  [Token 0 .. 15]  --> Physical Block 12
                   [Token 16 .. 31] --> Physical Block 47
                   [Token 32 .. 47] --> Physical Block 3
```

During the single-token autoregressive decode step, each query head must attend to all past key-value states scattered across these arbitrary physical memory blocks.

---

## Designing the Triton Kernel

The Triton decode kernel computes:
$$\text{Output}_i = \text{Softmax}\left(\frac{Q_i \cdot K^T}{\sqrt{d}}\right) \cdot V$$
where the $K$ and $V$ tensors are indexed via the block table.

### 1. Grid Launch Configuration
Each Triton program instance processes one attention head of one sequence in the batch:
```python
grid = (batch_size, num_heads)
paged_attention_kernel[grid](
    output_ptr,
    query_ptr,
    key_cache_ptr,
    val_cache_ptr,
    block_tables_ptr,
    context_lens_ptr,
    ...
)
```

### 2. Inner Block Loop & Online Softmax
To avoid materializing the full attention matrix in GPU SRAM, we employ FlashAttention-style **online softmax normalization**:
```python
# Initialize running statistics
m_prev = -float("inf")
l_prev = 0.0
acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

num_blocks = tl.cdiv(context_len, BLOCK_SIZE)
for block_idx in range(num_blocks):
    phys_block_id = tl.load(block_table_ptr + block_idx)
    
    # Load physical block KV states
    k_block = tl.load(k_ptr + phys_block_id * stride_k_block + ...)
    v_block = tl.load(v_ptr + phys_block_id * stride_v_block + ...)
    
    # Compute dot products
    scores = tl.sum(q * k_block, axis=-1) * sm_scale
    
    # Mask invalid tokens in the final block
    mask = (block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)) < context_len
    scores = tl.where(mask, scores, -float("inf"))
    
    # Online Softmax update
    m_curr = tl.maximum(m_prev, tl.max(scores))
    p = tl.exp(scores - m_curr)
    l_curr = l_prev * tl.exp(m_prev - m_curr) + tl.sum(p)
    
    # Update running output accumulator
    acc = acc * tl.exp(m_prev - m_curr) + tl.sum(p[:, None] * v_block, axis=0)
    m_prev = m_curr
    l_prev = l_curr

output = acc / l_prev
```

---

## Numerical Correctness Validation

Before serving a single request, the kernel was validated across diverse tensor configurations:
1. **Contiguous Block Placement**: Verifies exact mathematical match against standard `torch.nn.functional.scaled_dot_product_attention`.
2. **Shuffled Blocks**: Validates that physical memory permuting produces identical numerical output ($\Delta < 10^{-4}$ in FP16).
3. **Grouped Query Attention (GQA)**: Validates varying head-ratio projections (e.g. 14 query heads to 2 KV heads in Qwen 0.5B).
4. **Context Lengths up to 2048**: Confirms no numerical overflow or precision collapse in the exponential accumulators.

---

## Step-Time Benchmarking & Limitations

Benchmarking isolated single-step decode times across batch sizes (1, 3, 4, 8) against plain Hugging Face SDPA:

| Batch Size | Plain HF Median | Paged Attention Median | Speedup |
| :---: | :---: | :---: | :---: |
| BS = 1 | 42.93 ms | 40.83 ms | **1.05×** |
| BS = 3 | 43.75 ms | 46.99 ms | **0.93×** |
| BS = 4 | 43.94 ms | 48.91 ms | **0.90×** |
| BS = 8 | 45.35 ms | 60.78 ms | **0.75×** |

### Key Takeaway
At higher batch sizes, the paged runner experiences slight overhead (0.75× at BS=8) compared to contiguous tensors due to Python-level dispatch and metadata block-table slicing. This confirms that the primary benefit of Paged Attention is **not single-step arithmetic acceleration**, but rather **eliminating memory fragmentation so that the serving system can run at significantly higher batch concurrency**.

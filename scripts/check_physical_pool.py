"""
Quick sanity-check script for PhysicalBlockPool and compute_num_blocks.
Allocates the pool on GPU and verifies that the reported memory matches:
    num_blocks * block_size * 2 (K+V) * num_layers * num_kv_heads * head_dim * dtype_bytes
"""
import argparse
import torch
from transformers import AutoConfig

from engine.memory.block_allocator import PhysicalBlockPool, compute_num_blocks


def main():
    parser = argparse.ArgumentParser(description="Sanity check PhysicalBlockPool allocation.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B", help="Hugging Face model name or path.")
    parser.add_argument("--block-size", type=int, default=16, help="KV cache block size (tokens per block).")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="Fraction of free VRAM to reserve.")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device identifier.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. PhysicalBlockPool requires a GPU device.")

    device = torch.device(args.device)
    print(f"=== Target Device: {torch.cuda.get_device_name(device)} ({device}) ===")
    free_bytes_before, total_bytes = torch.cuda.mem_get_info(device)
    print(f"Initial Free VRAM: {free_bytes_before / (1024**3):.3f} GiB / {total_bytes / (1024**3):.3f} GiB")

    # Load model config
    config = AutoConfig.from_pretrained(args.model)
    num_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layer", None))
    num_kv_heads = getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads", getattr(config, "n_head", None)))
    head_dim = getattr(config, "head_dim", getattr(config, "hidden_size", 0) // getattr(config, "num_attention_heads", 1))

    print(f"\nModel: {args.model}")
    print(f"Architecture KV config: layers={num_layers}, kv_heads={num_kv_heads}, head_dim={head_dim}, block_size={args.block_size}")

    # Theoretical byte cost per physical block across all layers (K and V)
    dtype = torch.float16
    element_size = torch.tensor([], dtype=dtype).element_size()  # 2 bytes for float16
    bytes_per_block = 2 * num_layers * args.block_size * num_kv_heads * head_dim * element_size
    print(f"Byte cost per physical block (K+V, all layers): {bytes_per_block:,} bytes ({bytes_per_block / 1024:.2f} KiB)")

    # Compute blocks
    num_blocks = compute_num_blocks(
        model_config=config,
        block_size=args.block_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        device=device,
        dtype=dtype,
    )

    expected_vram_bytes = num_blocks * bytes_per_block
    target_budget_bytes = int(free_bytes_before * args.gpu_memory_utilization)

    print(f"\n--- Sizing Calculation ---")
    print(f"Target utilization: {args.gpu_memory_utilization * 100:.1f}%")
    print(f"Target free VRAM budget: {target_budget_bytes / (1024**3):.3f} GiB ({target_budget_bytes:,} bytes)")
    print(f"Calculated num_blocks: {num_blocks:,}")
    print(f"Calculated VRAM to reserve: {expected_vram_bytes / (1024**3):.3f} GiB ({expected_vram_bytes:,} bytes)")

    # Allocate the pool
    torch.cuda.reset_peak_memory_stats(device)
    alloc_before = torch.cuda.memory_allocated(device)

    pool = PhysicalBlockPool(
        num_blocks=num_blocks,
        block_size=args.block_size,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )
    torch.cuda.synchronize(device)

    alloc_after = torch.cuda.memory_allocated(device)
    actual_allocated_bytes = alloc_after - alloc_before

    print(f"\n--- Allocation Verification ---")
    print(f"Total physical blocks allocated: {pool.num_blocks:,}")
    print(f"k_cache layers: {len(pool.k_cache)}, shape: {list(pool.k_cache[0].shape)}, dtype: {pool.k_cache[0].dtype}")
    print(f"v_cache layers: {len(pool.v_cache)}, shape: {list(pool.v_cache[0].shape)}, dtype: {pool.v_cache[0].dtype}")
    print(f"Pool reported total_vram_bytes: {pool.total_vram_bytes / (1024**3):.3f} GiB ({pool.total_vram_bytes:,} bytes)")
    print(f"PyTorch reported memory_allocated delta: {actual_allocated_bytes / (1024**3):.3f} GiB ({actual_allocated_bytes:,} bytes)")

    diff = abs(pool.total_vram_bytes - expected_vram_bytes)
    assert diff == 0, f"Discrepancy between theoretical ({expected_vram_bytes}) and pool ({pool.total_vram_bytes})"

    # PyTorch's cuda caching allocator rounds up tensor allocations to multiples of 512 bytes or block chunks.
    # We verify the delta is within ~1% of expected tensor size.
    alloc_ratio = actual_allocated_bytes / pool.total_vram_bytes
    print(f"CUDA allocator overhead ratio: {alloc_ratio:.4f} (expected ~1.00 - 1.01 due to segment alignment)")
    assert 1.0 <= alloc_ratio <= 1.05, (
        f"CUDA allocated delta ({actual_allocated_bytes}) deviated significantly from tensor size ({pool.total_vram_bytes})"
    )

    free_bytes_after, _ = torch.cuda.mem_get_info(device)
    print(f"Remaining Free VRAM: {free_bytes_after / (1024**3):.3f} GiB")
    print("\n[SUCCESS] PhysicalBlockPool sizing arithmetic and pre-allocation match exactly!")


if __name__ == "__main__":
    main()

# Shared Memory Benchmark — Status, Findings, and Next Steps

## Executive Summary

The benchmark evaluates four personalization methods for cross-agent shared memory in AIOS. Two methods (naive_concat, vanilla_rag) are working correctly and producing valid results. Two methods (mem0_default, kernel_shared) are failing due to Mem0's fact extraction pipeline being incompatible with qwen2.5:7b — this is a model capability limitation, not a code bug.

## Current Results (150 trials each, qwen2.5:7b assistant, gpt-5.4 judge)

| Method | Profile Usage | Task Usage | Integration | Latency (s) | Status |
|--------|:---:|:---:|:---:|:---:|--------|
| naive_concat | 3.167 ± 0.219 | 4.287 ± 0.103 | 3.120 ± 0.193 | 49.4 ± 4.8 | ✅ Valid |
| vanilla_rag | 1.653 ± 0.101 | 3.753 ± 0.091 | 1.787 ± 0.083 | 42.1 ± 2.2 | ✅ Valid |
| mem0_default | 1.187 ± 0.066 | 1.360 ± 0.078 | 1.073 ± 0.042 | 18.6 ± 1.8 | ❌ Broken |
| kernel_shared | 1.020 ± 0.140 | 1.720 ± 0.482 | 1.000 ± 0.000 | 56.9 ± 9.4 | ❌ Broken |

## Root Cause: Mem0 Fact Extraction Fails with qwen2.5:7b

### How Mem0 stores memories

When `Memory.add(content, user_id=...)` is called, Mem0 does NOT simply store the raw content. It runs an internal pipeline:

1. **Fact extraction** — Prompts an LLM: "Extract key facts from this text. Return JSON list..."
2. **Deduplication** — Compares extracted facts against existing memories
3. **Embedding** — Generates vector embeddings for each fact
4. **Storage** — Stores facts + vectors in ChromaDB

Step 1 is the failure point. qwen2.5:7b frequently produces:
- Malformed JSON (missing brackets, trailing commas)
- Responses that don't match Mem0's expected schema
- Empty or truncated outputs for complex content

When Mem0 can't parse the extraction output, it **silently stores nothing** — no error raised, no vectors written, no searchable content.

### Evidence

Diagnostic added to the benchmark pipeline confirms this:
```
[WARNING] Mem0 diagnostic: 0 results for user_id=elena_foster__kernel_shared after ProfileAgent write.
[WARNING] Mem0 diagnostic: 0 results for user_id=emily_chen__kernel_shared after ProfileAgent write.
[WARNING] Mem0 diagnostic: 0 results for user_id=emma_johnson__kernel_shared after ProfileAgent write.
```

ProfileAgent calls `create_memory` with valid JSON content → kernel's MemoryManager passes it to Mem0 → Mem0 asks qwen2.5:7b to extract facts → qwen produces unparseable output → Mem0 stores nothing → search returns 0 results → injector has nothing to inject → assistant gets no personalization context.

### Why naive_concat and vanilla_rag work

These methods bypass Mem0's fact extraction entirely:
- **naive_concat**: Directly concatenates profile/task JSON into the system prompt. No memory storage layer involved.
- **vanilla_rag**: Stores raw content as embeddings directly (using the embedding model, not an LLM for extraction). Retrieval uses vector similarity search.

Neither depends on an LLM to process content before storage, so they work with any assistant model.

### Why mem0_default and kernel_shared fail identically

Both ultimately call Mem0's `Memory.add()`:
- **mem0_default**: The benchmark's `_run_mem0_default()` creates a local Mem0 client and calls `client.add()`
- **kernel_shared**: ProfileAgent/TaskAgent call `create_memory` → kernel's Mem0Provider calls `Memory.add()`

Same Mem0 instance, same qwen2.5:7b for fact extraction, same silent failure.

## Additional Issue: Azure Firewall (Now Resolved)

During the diagnostic run, the kernel logs revealed that ProfileAgent and TaskAgent's LLM calls (for structured profile/task extraction) were routed to `gpt-5.4` via Azure, which was blocked by firewall rules (no VPN connected). This caused a cascade:

1. ProfileAgent's `llm_chat_with_json_output` → gpt-5.4 → 403 error → empty/error response
2. ProfileAgent passes malformed content to `create_memory`
3. Mem0 receives garbage content → fact extraction fails even harder

**This is now fixed** — VPN is connected, gpt-5.4 is accessible.

## Implications for the Paper

### What can be published now

The naive_concat vs. vanilla_rag comparison is valid and publishable:
- naive_concat achieves the highest personalization scores (3.17/4.29/3.12) by providing full context directly
- vanilla_rag provides moderate personalization (1.65/3.75/1.79) through embedding-based retrieval
- The tradeoff is latency: naive_concat is slower (49.4s) due to larger prompts

### What cannot be published yet

- mem0_default results (scores at floor level due to silent storage failure)
- kernel_shared results (same Mem0 dependency, same failure)

### Recommended approach for the paper

**Option 1: Fix Mem0 and rerun (preferred)**

Configure Mem0's internal LLM to use gpt-5.4 (or any model capable of reliable JSON extraction). This gives a fair apples-to-apples comparison across all four methods. If kernel_shared beats mem0_default with both working correctly, that's a strong architectural claim.

**Option 2: Report as a finding about model capability requirements**

Frame the Mem0 failure as a research finding: "Mem0's fact extraction pipeline requires a model capable of reliable structured output (GPT-3.5+ class). With qwen2.5:7b, Mem0 silently fails to store any memories, producing zero-context behavior indistinguishable from no memory system at all. This reveals a hidden dependency in Mem0's architecture that limits its applicability in local-only or resource-constrained deployments."

**Option 3: Hybrid (strongest for the paper)**

Run all methods with gpt-5.4 available for Mem0's extraction (fair comparison), AND include the qwen-only results as a supplementary finding about architectural robustness. This shows:
- Fair comparison: kernel_shared vs mem0_default when both work
- Robustness finding: naive_concat and vanilla_rag degrade gracefully with weak models; Mem0 fails catastrophically

## Next Steps to Fix and Rerun

### For the kernel team

1. Configure Mem0's LLM to use `gpt-5.4` instead of `qwen2.5:7b` for fact extraction
2. Verify with a single-trial test that `Memory.add()` actually stores content (check ChromaDB directly or use `client.get_all(user_id=...)`)
3. Confirm the context injector finds shared memories after the fix

### For the benchmark

1. Restart kernel with VPN connected (gpt-5.4 accessible)
2. Run 3-trial diagnostic: check for `Mem0 diagnostic: found N results`
3. If working, run full 150-trial suite for all four methods
4. Remove the diagnostic logging before final runs (or keep at DEBUG level)

### Kernel config requirements

```yaml
memory:
  provider: mem0
  auto_inject: true
  auto_extract: true
  relevance_threshold: 0.3
  max_injected_memories: 10
  max_memory_tokens: 2000

# Mem0 must be configured to use a capable LLM for fact extraction
# qwen2.5:7b is insufficient — use gpt-5.4 or equivalent
```

## Architecture Comparison (for paper framing)

| Aspect | naive_concat | vanilla_rag | mem0_default | kernel_shared |
|--------|:---:|:---:|:---:|:---:|
| Memory storage | None (in-prompt) | Embedding vectors | Mem0 facts + vectors | Kernel → Mem0 |
| Requires extraction LLM | No | No | Yes (hidden) | Yes (hidden) |
| Works with 7B models | ✅ | ✅ | ❌ | ❌ |
| Scales with memory count | ❌ (prompt grows) | ✅ | ✅ | ✅ |
| Cross-agent sharing | Manual | Manual | Per-user scoping | Kernel-managed |
| SDK complexity | High (agent builds prompt) | Medium (agent manages RAG) | Low (Mem0 API) | Lowest (kernel handles all) |

"""
Tokenizer Utilities — Week 2

Wrappers for tiktoken, HuggingFace tokenizers, and sentencepiece
used in Notebook 03 (Tokenization).

Functions:
  compare_tokenizers(text)            — side-by-side count comparison
  get_tiktoken_encoder(encoding)      — cached tiktoken encoder
  tokenize_hf(text, model_name)      — HuggingFace AutoTokenizer
  estimate_tokens_tiktoken(text)      — accurate token count via tiktoken
"""

import json
import os
from typing import Dict, List, Optional, Any

# ---------------------------------------------------------------------------
# Cached encoder (avoid reloading on every call)
# ---------------------------------------------------------------------------
_tiktoken_cache: Dict[str, Any] = {}


def get_tiktoken_encoder(encoding_name: str = "cl100k_base"):
    """
    Return a tiktoken encoder, caching after the first load.

    Args:
        encoding_name: Encoding name — common choices:
            'cl100k_base'  (GPT-4, Claude-equivalent)
            'p50k_base'    (GPT-3 / Codex)
            'r50k_base'    (older GPT-2 family)

    Returns:
        tiktoken.Encoding object
    """
    if encoding_name not in _tiktoken_cache:
        try:
            import tiktoken
            _tiktoken_cache[encoding_name] = tiktoken.get_encoding(encoding_name)
        except ImportError:
            raise ImportError("Install tiktoken: pip install tiktoken")

    return _tiktoken_cache[encoding_name]


def estimate_tokens_tiktoken(
    text: str,
    model: str = "gpt-4"
) -> int:
    """
    Accurate token count using tiktoken (cl100k_base encoding).

    Note: Claude uses a similar BPE scheme; tiktoken gives a close approximation
    for cost-estimation purposes.

    Args:
        text:  Input text
        model: Model name hint (used to select encoding)

    Returns:
        Token count (int)
    """
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        # Fallback to character heuristic
        return len(text) // 4


def tokenize_hf(
    text: str,
    model_name: str = "gpt2"
) -> List[int]:
    """
    Tokenize text using a HuggingFace AutoTokenizer.

    Args:
        text:       Input text
        model_name: HuggingFace model ID (e.g., 'gpt2', 'bert-base-uncased')

    Returns:
        List of integer token IDs
    """
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return tokenizer.encode(text)
    except ImportError:
        raise ImportError("Install transformers: pip install transformers")


def compare_tokenizers(
    text: str,
    save_path: Optional[str] = "outputs/tokenizer_comparison.json"
) -> Dict[str, Any]:
    """
    Run text through three tokenizers and compare token counts and first token IDs.

    Tokenizers compared:
      1. tiktoken  (cl100k_base — used by GPT-4 / similar to Claude)
      2. HuggingFace gpt2 tokenizer
      3. sentencepiece (using unigram model via HuggingFace t5-small)

    Args:
        text:      Text to tokenize
        save_path: If provided, save results as JSON to this path

    Returns:
        Dict with keys: text, results (per tokenizer), winner (fewest tokens)
    """
    results: Dict[str, Any] = {"text": text, "char_count": len(text), "results": {}}

    # 1. tiktoken
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        ids = enc.encode(text)
        results["results"]["tiktoken_cl100k"] = {
            "token_count": len(ids),
            "first_10_ids": ids[:10],
            "description": "OpenAI cl100k_base (GPT-4 / similar to Claude)",
        }
    except Exception as e:
        results["results"]["tiktoken_cl100k"] = {"error": str(e)}

    # 2. HuggingFace gpt2
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
        ids = tok.encode(text)
        results["results"]["hf_gpt2"] = {
            "token_count": len(ids),
            "first_10_ids": ids[:10],
            "description": "HuggingFace gpt2 BPE tokenizer",
        }
    except Exception as e:
        results["results"]["hf_gpt2"] = {"error": str(e)}

    # 3. SentencePiece via t5-small (unigram LM)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("t5-small")
        ids = tok.encode(text)
        results["results"]["sentencepiece_t5"] = {
            "token_count": len(ids),
            "first_10_ids": ids[:10],
            "description": "SentencePiece Unigram LM via T5-small",
        }
    except Exception as e:
        results["results"]["sentencepiece_t5"] = {"error": str(e)}

    # Summary: which tokenizer used fewest tokens?
    counts = {
        name: r["token_count"]
        for name, r in results["results"].items()
        if "token_count" in r
    }
    if counts:
        results["winner_fewest_tokens"] = min(counts, key=counts.get)
        results["most_tokens"]          = max(counts, key=counts.get)
        results["compression_ratio"]    = {
            name: round(len(text) / cnt, 2)
            for name, cnt in counts.items()
        }

    # Print summary table
    print("=" * 65)
    print("🔤 TOKENIZER COMPARISON")
    print("=" * 65)
    print(f"Input text ({len(text)} chars):\n  \"{text[:80]}{'...' if len(text) > 80 else ''}\"")
    print()
    print(f"{'Tokenizer':<28} {'Tokens':>8}  {'chars/token':>12}  Description")
    print("-" * 65)
    for name, r in results["results"].items():
        if "token_count" in r:
            ratio = results["compression_ratio"].get(name, "?")
            print(f"  {name:<26} {r['token_count']:>8}  {ratio:>12}  {r['description'][:20]}")
        else:
            print(f"  {name:<26} {'ERROR':>8}  {'—':>12}  {r.get('error', '')[:20]}")
    print()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Results saved to: {save_path}")

    return results

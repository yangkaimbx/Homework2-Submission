"""
Attention Visualization Utilities — Week 2

Provides:
  - scaled_dot_product_attention: Pure NumPy implementation
  - plot_attention_heatmap: Matplotlib heatmap
  - multi_head_demo: BertViz wrapper (gracefully degrades to matplotlib)
  - MiniTransformerBlock: Single-block PyTorch Transformer for demonstration
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Scaled Dot-Product Attention (NumPy)
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute scaled dot-product attention.

        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    Args:
        Q: Query matrix  (seq_len, d_k)
        K: Key matrix    (seq_len, d_k)
        V: Value matrix  (seq_len, d_v)

    Returns:
        (output, attention_weights)
          - output:            (seq_len, d_v)
          - attention_weights: (seq_len, seq_len)  — rows sum to 1
    """
    d_k = Q.shape[-1]

    # Raw attention scores
    scores = Q @ K.T / np.sqrt(d_k)          # (seq_len, seq_len)

    # Softmax over last axis
    exp_scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attention_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    output = attention_weights @ V             # (seq_len, d_v)
    return output, attention_weights


# ---------------------------------------------------------------------------
# 2. Attention Heatmap (Matplotlib)
# ---------------------------------------------------------------------------

def plot_attention_heatmap(
    attention_weights: np.ndarray,
    tokens: List[str],
    title: str = "Attention Weights",
    save_path: Optional[str] = None
) -> None:
    """
    Plot an attention weight matrix as a colour heatmap.

    Args:
        attention_weights: (seq_len, seq_len) array — rows are queries, cols are keys
        tokens:            List of token strings (length = seq_len)
        title:             Plot title
        save_path:         If provided, save the figure to this path (PNG)
    """
    fig, ax = plt.subplots(figsize=(max(5, len(tokens)), max(4, len(tokens))))
    im = ax.imshow(attention_weights, cmap='viridis', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(tokens, fontsize=10)
    ax.set_xlabel("Keys (attended to)", fontsize=11)
    ax.set_ylabel("Queries (attending)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')

    # Annotate cells with weight values
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            val = attention_weights[i, j]
            colour = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                    color=colour, fontsize=8)

    plt.colorbar(im, ax=ax, label='Attention weight')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Heatmap saved to: {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# 3. Multi-Head Demo with BertViz (graceful degradation)
# ---------------------------------------------------------------------------

def multi_head_demo(
    sentence: str,
    model_name: str = "bert-base-uncased",
    save_dir: str = "outputs/attention_heatmaps",
    head_view: bool = True
) -> None:
    """
    Visualise multi-head attention for a sentence using BertViz.

    Falls back to a matplotlib head-averaged heatmap if BertViz is not installed
    or if the display environment does not support HTML widgets.

    Args:
        sentence:   Input sentence (keep it short — 8–15 tokens works best)
        model_name: HuggingFace model ID (default: bert-base-uncased)
        save_dir:   Directory to save fallback PNG heatmaps
        head_view:  If True, display BertViz head_view; False for model_view
    """
    os.makedirs(save_dir, exist_ok=True)

    try:
        from transformers import AutoTokenizer, AutoModel
        import torch

        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model     = AutoModel.from_pretrained(model_name, output_attentions=True)
        model.eval()

        inputs  = tokenizer(sentence, return_tensors='pt')
        tokens  = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        with torch.no_grad():
            outputs = model(**inputs)

        attentions = outputs.attentions  # tuple of (1, num_heads, seq, seq) per layer

        # Try BertViz first
        try:
            from bertviz import head_view as bv_head_view, model_view as bv_model_view
            print("✓ BertViz available — rendering interactive view")
            if head_view:
                bv_head_view(attentions, tokens)
            else:
                bv_model_view(attentions, tokens)
            return

        except ImportError:
            print("⚠ BertViz not installed — falling back to matplotlib heatmaps")

        # Fallback: plot head-averaged attention for each layer
        num_layers = len(attentions)
        for layer_idx in range(min(num_layers, 3)):        # show first 3 layers
            # average over heads
            avg_attn = attentions[layer_idx][0].mean(dim=0).numpy()  # (seq, seq)
            save_path = os.path.join(save_dir, f"layer{layer_idx+1:02d}_avg.png")
            plot_attention_heatmap(
                avg_attn, tokens,
                title=f"Layer {layer_idx+1} — Head-Averaged Attention\n\"{sentence}\"",
                save_path=save_path
            )

    except ImportError as e:
        print(f"❌ Could not load model: {e}")
        print("   Install: pip install transformers torch")


# ---------------------------------------------------------------------------
# 4. MiniTransformerBlock (PyTorch)
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class MiniTransformerBlock(nn.Module):
        """
        A single Transformer encoder block:
            MultiHeadSelfAttention → Add & Norm → FeedForward → Add & Norm

        Suitable for demonstrating the residual + LayerNorm pattern
        without needing a full model.

        Args:
            d_model:   Embedding dimension (default 64)
            num_heads: Number of attention heads (default 4)
            d_ff:      Feed-forward inner dimension (default 256)
            dropout:   Dropout rate (default 0.1)
        """

        def __init__(
            self,
            d_model:   int = 64,
            num_heads: int = 4,
            d_ff:      int = 256,
            dropout:   float = 0.1
        ):
            super().__init__()
            assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

            self.d_model   = d_model
            self.num_heads = num_heads
            self.d_head    = d_model // num_heads

            # Multi-head self-attention projections
            self.W_q = nn.Linear(d_model, d_model, bias=False)
            self.W_k = nn.Linear(d_model, d_model, bias=False)
            self.W_v = nn.Linear(d_model, d_model, bias=False)
            self.W_o = nn.Linear(d_model, d_model, bias=False)

            # Feed-forward network
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model),
            )

            # Layer normalisation (applied after residual)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

            self.dropout = nn.Dropout(dropout)

        def _split_heads(self, x: "torch.Tensor") -> "torch.Tensor":
            """(B, S, D) → (B, H, S, d_head)"""
            B, S, D = x.shape
            return x.view(B, S, self.num_heads, self.d_head).transpose(1, 2)

        def _merge_heads(self, x: "torch.Tensor") -> "torch.Tensor":
            """(B, H, S, d_head) → (B, S, D)"""
            B, H, S, d = x.shape
            return x.transpose(1, 2).contiguous().view(B, S, H * d)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            Args:
                x: (batch_size, seq_len, d_model)

            Returns:
                (batch_size, seq_len, d_model)
            """
            # --- Multi-head self-attention ---
            Q = self._split_heads(self.W_q(x))   # (B, H, S, d_head)
            K = self._split_heads(self.W_k(x))
            V = self._split_heads(self.W_v(x))

            scale  = self.d_head ** -0.5
            scores = (Q @ K.transpose(-2, -1)) * scale  # (B, H, S, S)
            attn   = F.softmax(scores, dim=-1)
            attn   = self.dropout(attn)

            out = self._merge_heads(attn @ V)            # (B, S, D)
            out = self.W_o(out)

            # --- Add & Norm (post-norm style) ---
            x = self.norm1(x + self.dropout(out))

            # --- Feed-forward ---
            ff_out = self.ff(x)
            x = self.norm2(x + self.dropout(ff_out))

            return x

        def count_parameters(self) -> int:
            """Return total number of trainable parameters."""
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

except ImportError:
    # torch not installed — provide a stub so imports don't fail
    class MiniTransformerBlock:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for MiniTransformerBlock. "
                "Install it with: pip install torch"
            )

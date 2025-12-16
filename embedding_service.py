from transformers import AutoTokenizer, AutoModel
import torch

EMBED_MODEL_ID = "/Volumes/HPLSSD/ai_models/embeddinggemma"
MAX_TOKENS = 512

_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
_model = AutoModel.from_pretrained(EMBED_MODEL_ID)
_model.eval()

# If available, use MPS/GPU for speed; otherwise CPU
_device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
_model.to(_device)

def embed_text(text: str):
    """Compute a 768-d embedding for the given text using the HF model; mean-pool last hidden state."""
    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TOKENS,
        padding=False,
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = _model(**inputs)
        last_hidden = outputs.last_hidden_state  # shape [1, L, H]
        emb = last_hidden.mean(dim=1).squeeze(0)  # shape [H]
        vec = emb.detach().cpu().numpy().astype(float).tolist()
        return vec
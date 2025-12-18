import os

from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import torch

from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker import HybridChunker

load_dotenv()

EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID")
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


def generate_chunks_for_about_shop(callback):
    source = Path('about_shop.md')

    # Pre-check: provide a clear error if the file is missing or name differs
    if not source.exists():
        raise FileNotFoundError(
            f"Input PDF not found at: {source}\n"
            "Tips: check the exact filename (including spaces/parentheses), and run the script from the project folder."
        )

    converter = DocumentConverter()
    result = converter.convert(source)
    doc = result.document

    print("Begin chunking...")

    chunk_tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_ID),
        max_tokens=MAX_TOKENS,  # optional, by default derived from `tokenizer` for HF case
    )

    chunker = HybridChunker(
        tokenizer=chunk_tokenizer,
        merge_peers=True,  # optional, defaults to True
    )

    chunk_iter = chunker.chunk(dl_doc=doc)
    chunks = list(chunk_iter)

    for chunk_index, chunk in enumerate(chunks):
        txt_tokens = _tokenizer.tokenize(chunk.text)

        ser_txt = chunker.contextualize(chunk=chunk)
        ser_tokens = _tokenizer.tokenize(ser_txt)
        callback(chunk_index, chunk.text, ser_txt, txt_tokens, ser_tokens)
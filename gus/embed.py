"""Embedding backends. The rest of Gus only calls embed_documents() and embed_query().

The model and device are chosen per project at `gus init` (stored in .gus/config.json); GUS_MODEL / GUS_DEVICE override.
"""

import sys
from functools import cache

import numpy as np

DEFAULT_MODEL = "minishlab/potion-base-8M"
QUALITY_MODEL = "onnx-community/embeddinggemma-300m-ONNX"
CHOICES = {"fast": DEFAULT_MODEL, "quality": QUALITY_MODEL}  # names offered at `gus init`; any model id works too
MODEL = DEFAULT_MODEL
DEVICE = "cpu"
MAX_TOKENS = 512
BATCH = 8  # EmbeddingGemma fp32 activations at 512 tokens are memory-hungry

# Transformer models run with onnxruntime (gus-mcp[onnx] on CPU, gus-mcp[gpu] on CUDA), measured on eval/. Any other model is
# loaded as a model2vec static model: much faster to index, less precise (see README, Embedding models).
ONNX_MODELS = {
    # fp32 on purpose: on CPU the q4 export is ~6x slower (4-bit weights are unpacked on every step) for no quality gain
    "onnx-community/embeddinggemma-300m-ONNX": dict(
        file="onnx/model.onnx", external_data=True, output="sentence_embedding", vector_weight=1.0,
        query_prefix="task: code retrieval | query: ", document_prefix="title: none | text: ",
    ),
}


def _vector_weight(model):
    # weight of the vector ranking against keyword ranking (1.0), chosen on eval dev: static embeddings are weaker on code
    return ONNX_MODELS.get(model, {}).get("vector_weight", 0.6)


VECTOR_WEIGHT = _vector_weight(MODEL)


def resolve(name: str) -> str:
    return CHOICES.get(name, name)


def use(model: str, device: str = "cpu") -> None:
    """Switch the active model and device ("cpu" or "gpu"; only transformer models use the GPU)."""
    global MODEL, DEVICE, VECTOR_WEIGHT
    model = resolve(model)
    if device not in ("cpu", "gpu"):
        raise SystemExit(f"gus: device must be 'cpu' or 'gpu', not {device!r}")
    if (model, device) != (MODEL, DEVICE):
        MODEL, DEVICE, VECTOR_WEIGHT = model, device, _vector_weight(model)
        _static.cache_clear()
        _onnx.cache_clear()


def missing_runtime(model: str, device: str) -> str | None:
    """Why this model/device can't run here, with the install command to fix it; None if it can."""
    if resolve(model) not in ONNX_MODELS:
        return None
    extra = "gpu" if device == "gpu" else "onnx"
    fix = f"install it with `uv tool install 'gus-mcp[{extra}]'` (or `pip install 'gus-mcp[{extra}]'`)"
    try:
        import onnxruntime
    except ImportError:
        return f"{resolve(model)} needs onnxruntime: {fix}"
    if device == "gpu" and "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
        return f"GPU needs the CUDA build of onnxruntime (and an NVIDIA GPU): {fix}; don't also install gus-mcp[onnx]"
    return None


def _download(repo, filename=None):
    from huggingface_hub import hf_hub_download, snapshot_download

    get = (lambda **kw: hf_hub_download(repo, filename, **kw)) if filename else (lambda **kw: snapshot_download(repo, **kw))
    try:  # don't touch the network when the model is already cached
        return get(local_files_only=True)
    except Exception:
        return get()


@cache
def _static():
    from model2vec import StaticModel

    return StaticModel.from_pretrained(_download(MODEL))


@cache
def _onnx():
    try:
        import onnxruntime
    except ImportError:
        raise SystemExit(f"gus: {missing_runtime(MODEL, DEVICE)}") from None
    from tokenizers import Tokenizer

    spec = ONNX_MODELS[MODEL]
    tok = Tokenizer.from_file(_download(MODEL, "tokenizer.json"))
    tok.enable_truncation(MAX_TOKENS)
    if spec.get("external_data"):
        _download(MODEL, spec["file"] + "_data")  # weights live next to the graph file
    providers = ["CPUExecutionProvider"]
    if DEVICE == "gpu":
        if hasattr(onnxruntime, "preload_dlls"):
            onnxruntime.preload_dlls()  # CUDA/cuDNN from the nvidia-* pip packages that gus-mcp[gpu] installs
        providers.insert(0, "CUDAExecutionProvider")
    session = onnxruntime.InferenceSession(_download(MODEL, spec["file"]), providers=providers)
    if DEVICE == "gpu" and session.get_providers()[0] != "CUDAExecutionProvider":
        # never slow down silently: a CPU fallback can turn a 2-minute index into half an hour
        why = missing_runtime(MODEL, DEVICE) or "the CUDA provider failed to load"
        print(f"gus: warning: GPU requested but not used ({why}); running on CPU", file=sys.stderr)
    return spec, tok, session


def _run_onnx(texts):
    spec, tok, session = _onnx()
    inputs = {i.name for i in session.get_inputs()}
    encodings = tok.encode_batch(texts)
    outputs = [o.name for o in session.get_outputs()]
    # models with a trained pooling/projection head export it as its own output; otherwise pool the hidden states
    pick = outputs.index(spec["output"]) if "output" in spec else 0
    out = np.zeros((len(texts), session.get_outputs()[pick].shape[-1]), dtype=np.float32)
    order = np.argsort([len(e.ids) for e in encodings])  # similar lengths per batch: less padding, much faster
    for b in range(0, len(order), BATCH):
        idx = order[b : b + BATCH]
        width = max(len(encodings[i].ids) for i in idx)
        ids = np.zeros((len(idx), width), dtype=np.int64)
        mask = np.zeros_like(ids)
        for row, i in enumerate(idx):
            n = len(encodings[i].ids)
            ids[row, :n], mask[row, :n] = encodings[i].ids, 1
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        result = session.run(None, feed)[pick]
        if "output" in spec:
            out[idx] = result
        elif spec["pool"] == "cls":
            out[idx] = result[:, 0]
        else:
            out[idx] = (result * mask[..., None]).sum(1) / mask.sum(1, keepdims=True)
    return out


def _embed(texts, prefix_key):
    if MODEL in ONNX_MODELS:
        prefix = ONNX_MODELS[MODEL].get(prefix_key, "")
        vecs = _run_onnx([prefix + t for t in texts])
    else:
        vecs = np.asarray(_static().encode(texts), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    return vecs


def embed_documents(texts: list[str]) -> np.ndarray:
    """L2-normalised float32 vectors, one row per chunk text."""
    return _embed(texts, "document_prefix")


def embed_query(texts: list[str]) -> np.ndarray:
    """L2-normalised float32 vectors for search queries (some models embed queries with an instruction prefix)."""
    return _embed(texts, "query_prefix")

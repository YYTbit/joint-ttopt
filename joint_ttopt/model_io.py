import json
from pathlib import Path

from .storage import record_digest


def tokenizer_digest(tokenizer):
    backend = getattr(tokenizer, "backend_tokenizer", None)
    backend_record = json.loads(backend.to_str()) if backend is not None else None
    if backend_record is not None:
        for field in ("padding", "truncation"):
            backend_record.pop(field, None)
    return record_digest({"vocabulary": tokenizer.get_vocab(),
                          "backend": backend_record,
                          "special_tokens": {name: str(value) for name, value in tokenizer.special_tokens_map.items()},
                          "chat_template": tokenizer.chat_template})


def quantization_options(config):
    quantization = getattr(config, "quantization_config", None)
    if hasattr(quantization, "to_dict"):
        quantization = quantization.to_dict()
    if quantization and quantization.get("quant_method") == "mxfp4":
        from transformers import Mxfp4Config
        return {"quantization_config": Mxfp4Config(dequantize=True)}
    return {}


def renderer_settings(options, model_type):
    result = dict(options)
    arguments = dict(result.get("chat_template_kwargs", {}))
    if model_type == "qwen3":
        arguments.setdefault("enable_thinking", False)
    if arguments:
        result["chat_template_kwargs"] = arguments
    return result


def policy_identity(policy):
    renderer = getattr(policy, "renderer", None)
    return {"model": policy.metadata,
            "temperature": getattr(policy, "temperature", None),
            "max_context": getattr(policy, "max_context", None),
            "renderer": renderer.configuration() if hasattr(renderer, "configuration") else None}


def entry_metadata(training_data):
    path = Path(training_data).parent / "metadata.json"
    if not path.is_file():
        return {}
    metadata = json.loads(path.read_text())
    if metadata.get("split") != "entry_training":
        raise ValueError("Action-head metadata requires the entry-training split")
    from .storage import file_digest
    if metadata.get("label_sha256") != file_digest(training_data):
        raise ValueError("Action-head labels differ from their collection metadata")
    return metadata

"""Model adapter compatibility exports."""

from .full_protocol import HuggingFaceSeq2SeqAdapter


def load_model(model_name: str):
    return HuggingFaceSeq2SeqAdapter(model_name)


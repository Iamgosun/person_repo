from __future__ import annotations

from train_py.families.base_family import BaseFamily
from train_py.families.deterministic_coop_family import DeterministicCoOpFamily
from train_py.families.text_only_bayes_coop_family import TextOnlyBayesCoOpFamily
from train_py.families.vlm_adapter_family import VLMAdapterFamily


def build_family(family_name: str) -> BaseFamily:
    key = str(family_name).strip().lower()
    if key == "deterministic_coop":
        return DeterministicCoOpFamily()
    if key == "text_only_bayes_coop":
        return TextOnlyBayesCoOpFamily()
    if key == "vlm_adapter":
        return VLMAdapterFamily()
    raise ValueError(f"unknown family: {family_name}")

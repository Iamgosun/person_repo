from __future__ import annotations

from train_py.recipes.base import BaseRecipe
from train_py.recipes.text_only_bayes_coop_recipe import TextOnlyBayesCoOpRecipe
from train_py.recipes.vlm_adapter_recipe import VLMAdapterRecipe


def build_recipe(method_name: str) -> BaseRecipe:
    key = str(method_name).strip().lower()

    if key == "text_only_bayes_coop":
        return TextOnlyBayesCoOpRecipe()

    if key == "vlm_adapter":
        return VLMAdapterRecipe()

    raise ValueError(f"未知 method_name: {method_name}")
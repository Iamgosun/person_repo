from __future__ import annotations

from train_py.recipes.base import BaseRecipe
from train_py.recipes.deterministic_coop_recipe import DeterministicCoOpRecipe
from train_py.recipes.text_only_bayes_coop_recipe import TextOnlyBayesCoOpRecipe
from train_py.recipes.vlm_adapter_recipe import VLMAdapterRecipe


def build_recipe(recipe_name: str) -> BaseRecipe:
    key = str(recipe_name).strip().lower()

    if key == "text_only_bayes_coop":
        return TextOnlyBayesCoOpRecipe()

    if key == "vlm_adapter":
        return VLMAdapterRecipe()

    if key in {"deterministic_coop", "deterministic_coop_standard"}:
        return DeterministicCoOpRecipe()

    raise ValueError(f"未知 recipe_name: {recipe_name}")

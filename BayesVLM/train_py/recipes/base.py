from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseRecipe(ABC):
    method_name: str = ""
    best_checkpoint_filename: str = "best_model.pt"
    require_image_feature_cache: bool = False

    @abstractmethod
    def run_path_parts(self, args) -> list[str]:
        raise NotImplementedError

    def validate_and_note(self, args) -> None:
        pass

    @abstractmethod
    def build_state(self, ctx, args) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def build_config_extra(self, state: dict[str, Any], ctx, args) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def train_one_epoch(self, state: dict[str, Any], ctx, args, epoch: int) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def evaluate_split(self, state: dict[str, Any], loader, ctx, args) -> dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def format_epoch_log(self, row: dict[str, Any], ctx, args) -> str:
        raise NotImplementedError

    @abstractmethod
    def build_best_state(
        self,
        state: dict[str, Any],
        ctx,
        args,
        epoch: int,
        val_metrics: dict[str, float],
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def load_best_state(self, state: dict[str, Any], best_state: dict[str, Any], ctx, args) -> None:
        raise NotImplementedError

    @abstractmethod
    def dump_predictions(self, state: dict[str, Any], ctx, args) -> None:
        raise NotImplementedError

    def build_summary_extra(
        self,
        state: dict[str, Any],
        best_state: dict[str, Any],
        ctx,
        args,
    ) -> dict[str, Any]:
        return {}
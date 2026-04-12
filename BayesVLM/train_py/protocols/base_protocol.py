from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ProtocolEvalSplit:
    split_name: str
    loader_attr: str
    class_names_attr: str
    relative_output_dir: str


class BaseProtocol(ABC):
    protocol_name: str = ""

    @abstractmethod
    def prepare_train_data(self, args):
        raise NotImplementedError

    @abstractmethod
    def prepare_eval_data(self, args):
        raise NotImplementedError

    @abstractmethod
    def classification_splits(self, ctx, args) -> list[ProtocolEvalSplit]:
        raise NotImplementedError

    def build_summary_extra(self, ctx, args) -> dict:
        del ctx, args
        return {}

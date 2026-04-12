from __future__ import annotations

from train_py.protocols.base_protocol import ProtocolEvalSplit
from train_py.protocols.xd_protocol import XDProtocol


class DGProtocol(XDProtocol):
    protocol_name = "dg"

    def classification_splits(self, ctx, args):
        splits = [ProtocolEvalSplit("source_test", "test_loader", "class_names", "eval/protocol/dg/source/test")]
        for target_dataset in getattr(args, "target_datasets", []):
            key = f"target_{target_dataset}"
            splits.append(
                ProtocolEvalSplit(
                    f"target_{target_dataset}",
                    f"extra_eval_loaders.{key}",
                    f"extra_eval_class_names.{key}",
                    f"eval/protocol/dg/targets/{target_dataset}",
                )
            )
        return splits

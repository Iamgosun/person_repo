from __future__ import annotations

from bayesvlm.data.class_split import build_base2new_split
from bayesvlm.data.dataset_ops import build_fewshot_subset
from bayesvlm.data.pipeline import PreparedDataBundle, prepare_raw_data_bundle
from bayesvlm.utils import get_image_size, get_model_type_and_size, get_transform
from train_py.protocols.base_protocol import BaseProtocol, ProtocolEvalSplit


class Base2NewProtocol(BaseProtocol):
    protocol_name = "base2new"

    def _eval_transform(self, args):
        model_type, _ = get_model_type_and_size(args.model)
        image_size = get_image_size(args.model)
        return get_transform(model_type, image_size)

    def _build_bundle(self, args):
        eval_transform = self._eval_transform(args)
        raw = prepare_raw_data_bundle(
            dataset=args.dataset,
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_transform=eval_transform,
            test_transform=eval_transform,
            shuffle_train=True,
        )
        split = build_base2new_split(raw.train_ds, raw.val_ds, raw.test_ds)
        train_ds = build_fewshot_subset(split.base_train_full, shots_per_class=args.shots_per_class, seed=args.seed, strict=True)
        extras = {"new_test": (split.new_test, split.new_class_names)}
        for ood_dataset in getattr(args, "ood_datasets", []):
            raw_ood = prepare_raw_data_bundle(
                dataset=ood_dataset,
                data_root=args.data_root,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                train_transform=eval_transform,
                test_transform=eval_transform,
                shuffle_train=False,
            )
            extras[ood_dataset] = (raw_ood.test_ds, raw_ood.class_names)
        return PreparedDataBundle(
            dataset_name=args.dataset,
            train_ds=train_ds,
            train_eval_ds=train_ds,
            val_ds=split.base_val,
            test_ds=split.base_test,
            class_names=split.base_class_names,
            extra_eval_datasets=extras,
            metadata={"base_labels": split.base_labels, "new_labels": split.new_labels},
        )

    def prepare_train_data(self, args):
        return self._build_bundle(args)

    def prepare_eval_data(self, args):
        return self._build_bundle(args)

    def classification_splits(self, ctx, args):
        del args
        return [
            ProtocolEvalSplit("base_train", "train_eval_loader", "class_names", "eval/protocol/base2new/base/train"),
            ProtocolEvalSplit("base_val", "val_loader", "class_names", "eval/protocol/base2new/base/val"),
            ProtocolEvalSplit("base_test", "test_loader", "class_names", "eval/protocol/base2new/base/test"),
            ProtocolEvalSplit("new_test", "extra_eval_loaders.new_test", "extra_eval_class_names.new_test", "eval/protocol/base2new/new/test"),
        ]

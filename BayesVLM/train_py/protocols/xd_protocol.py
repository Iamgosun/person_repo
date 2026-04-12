from __future__ import annotations

from bayesvlm.data.pipeline import build_id_prepared_bundle, prepare_raw_data_bundle
from bayesvlm.utils import get_image_size, get_model_type_and_size, get_transform
from train_py.protocols.base_protocol import BaseProtocol, ProtocolEvalSplit


class XDProtocol(BaseProtocol):
    protocol_name = "xd"

    def _eval_transform(self, args):
        model_type, _ = get_model_type_and_size(args.model)
        image_size = get_image_size(args.model)
        return get_transform(model_type, image_size)

    def _build_bundle(self, args):
        eval_transform = self._eval_transform(args)
        source_raw = prepare_raw_data_bundle(
            dataset=args.dataset,
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_transform=eval_transform,
            test_transform=eval_transform,
            shuffle_train=True,
        )
        prepared = build_id_prepared_bundle(source_raw, dataset_name=args.dataset, shots_per_class=args.shots_per_class, seed=args.seed)
        extras = {}
        for target_dataset in getattr(args, "target_datasets", []):
            raw = prepare_raw_data_bundle(
                dataset=target_dataset,
                data_root=args.data_root,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                train_transform=eval_transform,
                test_transform=eval_transform,
                shuffle_train=False,
            )
            if list(raw.class_names) != list(prepared.class_names):
                raise ValueError(f"xd currently requires identical label spaces. source={args.dataset} target={target_dataset}")
            extras[f"target_{target_dataset}"] = (raw.test_ds, raw.class_names)
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
        prepared.extra_eval_datasets = extras
        return prepared

    def prepare_train_data(self, args):
        return self._build_bundle(args)

    def prepare_eval_data(self, args):
        return self._build_bundle(args)

    def classification_splits(self, ctx, args):
        splits = [ProtocolEvalSplit("source_test", "test_loader", "class_names", "eval/protocol/xd/source/test")]
        for target_dataset in getattr(args, "target_datasets", []):
            key = f"target_{target_dataset}"
            splits.append(ProtocolEvalSplit(
                f"target_{target_dataset}",
                f"extra_eval_loaders.{key}",
                f"extra_eval_class_names.{key}",
                f"eval/protocol/xd/targets/{target_dataset}",
            ))
        return splits

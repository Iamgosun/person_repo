from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

BOOL_KEYS = {
    "csc",
    "nesterov",
    "rebuild_image_feature_cache",
    "disable_cache_image_features",
    "use_data_augmentation",
    "use_augmented_train_cache",
    "use_full_cov",
    "save_prototype_history",
    "train_logit_scale",
}

INT_KEYS = {
    "shots_per_class",
    "seed",
    "n_ctx",
    "epochs",
    "batch_size",
    "num_workers",
    "prediction_topk",
    "warmup_epoch",
    "train_aug_repeats",
    "pseudo_data_count",
    "lambda_opt_steps",
    "hybrid_warmup_epochs",
    "clipa_hidden_dim",
    "gaussian_mc_samples",
    "gaussian_anneal_start_epoch",
    "bayesadapter_train_mc_samples",
    "bayesadapter_eval_mc_samples",
}

FLOAT_KEYS = {
    "lr",
    "weight_decay",
    "momentum",
    "warmup_cons_lr",
    "lambda_txt_init",
    "map_loss_weight",
    "bayes_loss_weight",
    "ctx_reg_weight",
    "taskres_alpha",
    "clipa_ratio",
    "tipa_alpha",
    "tipa_beta",
    "gaussian_prior_sigma",
    "bayesadapter_prior_sigma",
    "bayesadapter_kl_scale_divisor",
    "bayesadapter_text_only_mu_blend_lambda",
}

LIST_INT_KEYS = {
    "prototype_track_class_ids",
}

LIST_STR_KEYS = {
    "evaluation_tasks",
    "target_datasets",
    "ood_datasets",
    "ood_scores",
}

COMMON_REQUIRED_KEYS = {
    "model",
    "local_model_path",
    "data_root",
    "save_dir",
    "image_feature_cache_root",
    "optimizer",
    "lr",
    "weight_decay",
    "epochs",
    "lr_scheduler",
    "warmup_epoch",
    "warmup_cons_lr",
    "model_selection",
    "selection_metric",
    "selection_mode",
    "device",
    "batch_size",
    "num_workers",
    "prediction_topk",
    "rebuild_image_feature_cache",
    "disable_cache_image_features",
    "use_data_augmentation",
    "use_augmented_train_cache",
    "train_aug_repeats",
}

EXPERIMENT_REQUIRED_KEYS = {
    "family",
    "variant",
    "protocol",
    "evaluation_tasks",
    "dataset",
    "shots_per_class",
    "seed",
}

FAMILY_REQUIRED_KEYS: dict[str, set[str]] = {
    "deterministic_coop": {
        "n_ctx",
        "ctx_init",
        "csc",
        "class_token_position",
    },
    "text_only_bayes_coop": {
        "n_ctx",
        "ctx_init",
        "csc",
        "class_token_position",
        "hessian_dir",
        "pseudo_data_count",
        "lambda_txt_init",
        "lambda_opt_steps",
        "use_full_cov",
        "train_objective",
        "hybrid_warmup_epochs",
        "map_loss_weight",
        "bayes_loss_weight",
        "ctx_reg_weight",
        "save_prototype_history",
        "prototype_track_class_ids",
    },
    "vlm_adapter": {
        "initialization",
    },
}

VLM_VARIANT_REQUIRED_KEYS: dict[str, set[str]] = {
    "LP": set(),
    "TR": {"taskres_alpha"},
    "CLIPA": {"clipa_ratio", "clipa_hidden_dim"},
    "TIPA": {"tipa_alpha", "tipa_beta"},
    "CROSSMODAL": set(),
    "GAUSSIAN_PER_CLASS": {
        "gaussian_prior_sigma",
        "gaussian_mc_samples",
        "gaussian_anneal_start_epoch",
    },
    "BAYESADAPTER": {
        "bayesadapter_prior_sigma",
        "bayesadapter_train_mc_samples",
        "bayesadapter_eval_mc_samples",
        "bayesadapter_kl_scale_divisor",
        "bayesadapter_covariance_mode",
        "bayesadapter_text_only_ckpt",
        "bayesadapter_text_only_run_dir_template",
        "bayesadapter_text_only_mu_strategy",
        "bayesadapter_text_only_sigma_strategy",
    },
}

PROTOCOL_REQUIRED_KEYS: dict[str, set[str]] = {
    "id": set(),
    "base2new": set(),
    "xd": {"target_datasets"},
    "dg": {"target_datasets"},
}

TASK_REQUIRED_KEYS: dict[str, set[str]] = {
    "classification": set(),
    "ood_detection": {"ood_datasets"},
}


@dataclass
class ExperimentPlan:
    plan_path: Path
    base_config: dict[str, Any]
    experiments: list[dict[str, Any]]


def _read_text(elem: ET.Element) -> str:
    return "" if elem.text is None else elem.text.strip()


def _parse_simple_element(elem: ET.Element) -> Any:
    children = list(elem)
    if not children:
        return _read_text(elem)
    if all(child.tag == "item" for child in children):
        return [_parse_simple_element(child) for child in children]
    result: dict[str, Any] = {}
    for child in children:
        if child.tag in result:
            raise ValueError(f"Duplicate XML tag <{child.tag}> under <{elem.tag}>")
        result[child.tag] = _parse_simple_element(child)
    return result


def _resolve_import_path(current_file: Path, href: str) -> Path:
    return (current_file.parent / href).resolve()


def _merge_layers(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    merged.update(override)
    return merged


def _load_config_fragment_recursive(path: Path, active_stack: tuple[Path, ...]) -> dict[str, Any]:
    path = path.resolve()
    if path in active_stack:
        cycle = " -> ".join(str(p) for p in [*active_stack, path])
        raise ValueError(f"cyclic XML import detected: {cycle}")

    root = ET.parse(path).getroot()
    if root.tag != "config":
        raise ValueError(f"{path} must have root <config>, got <{root.tag}>")

    merged: dict[str, Any] = {}
    imports_elem = root.find("imports")
    if imports_elem is not None:
        for import_elem in imports_elem.findall("import"):
            href = import_elem.attrib.get("href", "").strip()
            if not href:
                raise ValueError(f"{path} contains <import> without href")
            child_cfg = _load_config_fragment_recursive(
                _resolve_import_path(path, href),
                (*active_stack, path),
            )
            merged = _merge_layers(merged, child_cfg)

    for child in root:
        if child.tag == "imports":
            continue
        merged[child.tag] = _parse_simple_element(child)
    return merged


def load_config_fragment(path: str | Path) -> dict[str, Any]:
    return _load_config_fragment_recursive(Path(path), tuple())


def _parse_experiment_element(elem: ET.Element) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for child in elem:
        if child.tag in result:
            raise ValueError(f"duplicate key in <experiment>: {child.tag}")
        result[child.tag] = _parse_simple_element(child)
    return result


def _load_experiments_fragment(path: Path) -> list[dict[str, Any]]:
    path = path.resolve()
    root = ET.parse(path).getroot()
    if root.tag != "experiments":
        raise ValueError(f"{path} must have root <experiments>, got <{root.tag}>")
    experiments = []
    for child in root:
        if child.tag != "experiment":
            raise ValueError(f"{path}: only <experiment> is allowed under <experiments>")
        experiments.append(_parse_experiment_element(child))
    if not experiments:
        raise ValueError(f"{path} contains no experiments")
    return experiments


def load_experiment_plan(path: str | Path) -> ExperimentPlan:
    plan_path = Path(path).resolve()
    root = ET.parse(plan_path).getroot()
    if root.tag not in {"experiment_plan", "plan"}:
        raise ValueError(f"{plan_path} must have root <experiment_plan> or <plan>")

    base_config: dict[str, Any] = {}
    experiments: list[dict[str, Any]] = []

    imports_elem = root.find("imports")
    if imports_elem is not None:
        for import_elem in imports_elem.findall("import"):
            href = import_elem.attrib.get("href", "").strip()
            if not href:
                raise ValueError(f"{plan_path} has <import> without href")
            base_config = _merge_layers(
                base_config,
                load_config_fragment(_resolve_import_path(plan_path, href)),
            )

    for child in root:
        if child.tag in {"imports", "experiments"}:
            continue
        base_config[child.tag] = _parse_simple_element(child)

    experiments_elem = root.find("experiments")
    if experiments_elem is None:
        raise ValueError(f"{plan_path} is missing <experiments>")

    for child in experiments_elem:
        if child.tag == "experiment":
            experiments.append(_parse_experiment_element(child))
        elif child.tag == "import":
            href = child.attrib.get("href", "").strip()
            if not href:
                raise ValueError(f"{plan_path}: experiment import missing href")
            experiments.extend(_load_experiments_fragment(_resolve_import_path(plan_path, href)))
        else:
            raise ValueError(f"{plan_path}: invalid tag under <experiments>: <{child.tag}>")

    if not experiments:
        raise ValueError(f"{plan_path} contains no experiments")
    return ExperimentPlan(plan_path=plan_path, base_config=base_config, experiments=experiments)


def _to_bool(key: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"{key} must be true/false, got: {value}")


def _coerce_scalar(key: str, value: Any) -> Any:
    if key in BOOL_KEYS:
        return _to_bool(key, value)
    if key in INT_KEYS:
        return int(str(value).strip())
    if key in FLOAT_KEYS:
        return float(str(value).strip())
    return value


def _coerce_value(key: str, value: Any) -> Any:
    if key in LIST_INT_KEYS:
        if not isinstance(value, list):
            raise ValueError(f"{key} must be an <item> list")
        return [int(str(x).strip()) for x in value if str(x).strip()]
    if key in LIST_STR_KEYS:
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]
        text = str(value).strip()
        return [text] if text else []
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, dict):
        raise ValueError(
            f"Config fragments must remain flat. Key {key} resolved to a nested object."
        )
    return _coerce_scalar(key, value)


def coerce_flat_config(raw_cfg: dict[str, Any]) -> dict[str, Any]:
    return {key: _coerce_value(key, value) for key, value in raw_cfg.items()}


def _require_keys(cfg: dict[str, Any], required: set[str], where: str) -> None:
    missing = sorted(k for k in required if k not in cfg)
    if missing:
        raise ValueError(f"{where} missing required keys: {missing}")


def _validate_value_ranges(cfg: dict[str, Any]) -> None:
    if int(cfg["epochs"]) < 0:
        raise ValueError("epochs must be >= 0")
    if int(cfg["batch_size"]) <= 0:
        raise ValueError("batch_size must be > 0")
    if int(cfg["num_workers"]) < 0:
        raise ValueError("num_workers must be >= 0")
    if int(cfg["prediction_topk"]) <= 0:
        raise ValueError("prediction_topk must be > 0")
    if int(cfg["shots_per_class"]) < 0:
        raise ValueError("shots_per_class must be >= 0")
    if float(cfg["lr"]) <= 0:
        raise ValueError("lr must be > 0")
    if float(cfg["weight_decay"]) < 0:
        raise ValueError("weight_decay must be >= 0")
    if "bayesadapter_text_only_mu_blend_lambda" in cfg:
        lam = float(cfg["bayesadapter_text_only_mu_blend_lambda"])
        if not (0.0 <= lam <= 1.0):
            raise ValueError("bayesadapter_text_only_mu_blend_lambda must be in [0, 1]")


def _validate_enums(cfg: dict[str, Any]) -> None:
    if str(cfg["optimizer"]).lower() not in {"sgd", "adam", "adamw"}:
        raise ValueError("optimizer must be one of ['sgd', 'adam', 'adamw']")
    if str(cfg["lr_scheduler"]).lower() not in {"none", "cosine"}:
        raise ValueError("lr_scheduler must be one of ['none', 'cosine']")
    if str(cfg["model_selection"]).lower() not in {"best", "last"}:
        raise ValueError("model_selection must be one of ['best', 'last']")
    if str(cfg["selection_metric"]).lower() not in {"loss", "acc", "nlpd", "ece"}:
        raise ValueError("selection_metric must be one of ['loss', 'acc', 'nlpd', 'ece']")
    if str(cfg["selection_mode"]).lower() not in {"auto", "min", "max"}:
        raise ValueError("selection_mode must be one of ['auto', 'min', 'max']")
    if str(cfg["family"]).lower() not in FAMILY_REQUIRED_KEYS:
        raise ValueError(f"unknown family: {cfg['family']}")
    if str(cfg["protocol"]).lower() not in PROTOCOL_REQUIRED_KEYS:
        raise ValueError(f"unknown protocol: {cfg['protocol']}")
    for task in cfg["evaluation_tasks"]:
        if task not in TASK_REQUIRED_KEYS:
            raise ValueError(f"unknown evaluation task: {task}")
    if str(cfg["family"]).lower() == "vlm_adapter":
        if str(cfg["variant"]).upper() not in VLM_VARIANT_REQUIRED_KEYS:
            raise ValueError(f"unknown vlm_adapter variant: {cfg['variant']}")
    if "train_objective" in cfg:
        if str(cfg["train_objective"]).lower() not in {"map", "bayes", "hybrid"}:
            raise ValueError("train_objective must be one of ['map', 'bayes', 'hybrid']")
    if "bayesadapter_covariance_mode" in cfg:
        if str(cfg["bayesadapter_covariance_mode"]).lower() not in {"paper_scalar", "diag"}:
            raise ValueError("bayesadapter_covariance_mode must be one of ['paper_scalar', 'diag']")
    if "bayesadapter_text_only_mu_strategy" in cfg:
        if str(cfg["bayesadapter_text_only_mu_strategy"]).lower() not in {"replace", "blend"}:
            raise ValueError("bayesadapter_text_only_mu_strategy must be one of ['replace', 'blend']")
    if "bayesadapter_text_only_sigma_strategy" in cfg:
        if str(cfg["bayesadapter_text_only_sigma_strategy"]).lower() not in {"ignore", "override"}:
            raise ValueError("bayesadapter_text_only_sigma_strategy must be one of ['ignore', 'override']")
    if "ood_reference_split" in cfg and not str(cfg["ood_reference_split"]).strip():
        raise ValueError("ood_reference_split must not be empty when provided")


def _has_text_only_bridge(cfg: dict[str, Any]) -> bool:
    run_dir = str(cfg.get("bayesadapter_text_only_run_dir", "")).strip()
    run_dir_template = str(cfg.get("bayesadapter_text_only_run_dir_template", "")).strip()
    return bool(run_dir or run_dir_template)


def materialize_derived_fields(cfg: dict[str, Any]) -> dict[str, Any]:
    result = dict(cfg)
    result["family"] = str(result["family"]).strip().lower()
    result["protocol"] = str(result["protocol"]).strip().lower()
    if result["family"] == "vlm_adapter":
        result["variant"] = str(result["variant"]).strip().upper()
    else:
        result["variant"] = str(result.get("variant", "default") or "default").strip().lower()
    result["evaluation_tasks"] = [str(x).strip().lower() for x in result.get("evaluation_tasks", []) if str(x).strip()]
    if not result["evaluation_tasks"]:
        result["evaluation_tasks"] = ["classification"]
    if "ood_detection" in result["evaluation_tasks"]:
        result.setdefault("ood_scores", ["msp", "entropy", "energy"])
        default_ref = {
            "id": "test",
            "base2new": "base_test",
            "xd": "source_test",
            "dg": "source_test",
        }.get(result["protocol"], "test")
        result.setdefault("ood_reference_split", default_ref)
    result["cache_image_features"] = not bool(result["disable_cache_image_features"])
    result["output_name"] = str(result.get("output_name", result["family"])).strip()
    if result["family"] == "vlm_adapter" and result["variant"] == "BAYESADAPTER":
        if "bayesadapter_text_only_run_dir" not in result and "bayesadapter_text_only_run_dir_template" in result:
            template = str(result["bayesadapter_text_only_run_dir_template"])
            format_env = {
                "save_dir": result["save_dir"],
                "dataset": result["dataset"],
                "shots_per_class": result["shots_per_class"],
                "seed": result["seed"],
                "family": result["family"],
                "variant": result["variant"],
                "output_name": result["output_name"],
                "protocol": result["protocol"],
            }
            try:
                result["bayesadapter_text_only_run_dir"] = template.format(**format_env)
            except KeyError as exc:
                raise ValueError(
                    f"bayesadapter_text_only_run_dir_template uses unknown placeholder: {exc}"
                ) from exc
    return result


def validate_final_config(cfg: dict[str, Any]) -> None:
    _require_keys(cfg, COMMON_REQUIRED_KEYS, "common config")
    _require_keys(cfg, EXPERIMENT_REQUIRED_KEYS, "experiment config")
    _require_keys(cfg, FAMILY_REQUIRED_KEYS[cfg["family"]], f"family={cfg['family']}")
    if cfg["family"] == "vlm_adapter":
        _require_keys(cfg, VLM_VARIANT_REQUIRED_KEYS[cfg["variant"]], f"variant={cfg['variant']}")
        if cfg["variant"] == "BAYESADAPTER" and _has_text_only_bridge(cfg):
            _require_keys(
                cfg,
                {
                    "bayesadapter_text_only_ckpt",
                    "bayesadapter_text_only_mu_strategy",
                    "bayesadapter_text_only_sigma_strategy",
                    "bayesadapter_text_only_run_dir",
                },
                "bayesadapter text-only bridge",
            )
            if str(cfg["bayesadapter_text_only_mu_strategy"]).strip().lower() == "blend":
                _require_keys(cfg, {"bayesadapter_text_only_mu_blend_lambda"}, "bayesadapter blend strategy")
    _require_keys(cfg, PROTOCOL_REQUIRED_KEYS[cfg["protocol"]], f"protocol={cfg['protocol']}")
    for task in cfg["evaluation_tasks"]:
        _require_keys(cfg, TASK_REQUIRED_KEYS[task], f"evaluation_task={task}")
    _validate_enums(cfg)
    _validate_value_ranges(cfg)


def build_resolved_run_dicts(plan_path: str | Path) -> list[dict[str, Any]]:
    plan = load_experiment_plan(plan_path)
    base_config = coerce_flat_config(plan.base_config)
    run_dicts: list[dict[str, Any]] = []
    for idx, experiment_raw in enumerate(plan.experiments, start=1):
        experiment_cfg = coerce_flat_config(experiment_raw)
        merged = _merge_layers(base_config, experiment_cfg)
        merged = materialize_derived_fields(merged)
        validate_final_config(merged)
        merged["_experiment_index"] = idx
        run_dicts.append(merged)
    return run_dicts


def build_resolved_run_namespaces(plan_path: str | Path) -> list[Namespace]:
    return [Namespace(**cfg) for cfg in build_resolved_run_dicts(plan_path)]

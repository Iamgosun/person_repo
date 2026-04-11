from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET


# ============================================================
# 1) 基础类型声明
# ============================================================
def _has_text_only_bridge(cfg: dict[str, Any]) -> bool:
    run_dir = str(cfg.get("bayesadapter_text_only_run_dir", "")).strip()
    run_dir_template = str(cfg.get("bayesadapter_text_only_run_dir_template", "")).strip()
    return bool(run_dir or run_dir_template)


def _reject_legacy_bayesadapter_text_only_keys(cfg: dict[str, Any]) -> None:
    legacy = sorted(k for k in LEGACY_BAYESADAPTER_TEXT_ONLY_KEYS if k in cfg)
    if legacy:
        raise ValueError(
            "以下旧参数已废弃，请改用新的 bayesadapter_text_only_* 参数："
            f"{legacy}"
        )


BOOL_KEYS = {
    "csc",
    "nesterov",
    "rebuild_image_feature_cache",
    "disable_cache_image_features",
    "use_data_augmentation",
    "use_augmented_train_cache",
    "use_full_cov",
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


# ============================================================
# 2) 必填字段规则
# ============================================================

# 真正的全局公共字段：所有任务都需要
REQUIRED_COMMON_KEYS = {
    "model",
    "local_model_path",
    "data_root",
    "save_dir",
    "image_feature_cache_root",
    "lr",
    "weight_decay",
    "epochs",
    "optimizer",
    "momentum",
    "nesterov",
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

# 每条 experiment 都必须提供
REQUIRED_EXPERIMENT_KEYS = {
    "dataset",
    "shots_per_class",
    "seed",
}

# recipe 级别必填
RECIPE_REQUIRED_KEYS: dict[str, set[str]] = {
    "text_only_bayes_coop": {
        "recipe_name",
        "method_name",
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
        "prototype_track_class_ids",
    },
    "deterministic_coop": {
        "recipe_name",
        "method_name",
        "n_ctx",
        "ctx_init",
        "csc",
        "class_token_position",
    },
    "vlm_adapter": {
        "recipe_name",
        "method_name",
        "adapter_name",
        "initialization",
        # 这两个字段当前 recipe 会写入 config / note，保留显式要求更稳
        "hessian_dir",
        "pseudo_data_count",
    },
}

# adapter-specific 必填
TR_REQUIRED_KEYS = {
    "taskres_alpha",
}

CLIPA_REQUIRED_KEYS = {
    "clipa_ratio",
    "clipa_hidden_dim",
}

TIPA_REQUIRED_KEYS = {
    "tipa_alpha",
    "tipa_beta",
}

GAUSSIAN_PER_CLASS_REQUIRED_KEYS = {
    "gaussian_prior_sigma",
    "gaussian_mc_samples",
    "gaussian_anneal_start_epoch",
}


TEXT_ONLY_PRIOR_REQUIRED_KEYS = {
    "text_only_method_name",
    "bayesadapter_text_only_ckpt",
    "bayesadapter_text_only_run_dir_template",
}

BAYESADAPTER_REQUIRED_KEYS = {
    "bayesadapter_prior_sigma",
    "bayesadapter_train_mc_samples",
    "bayesadapter_eval_mc_samples",
    "bayesadapter_kl_scale_divisor",
    "bayesadapter_covariance_mode",
}

TEXT_ONLY_BRIDGE_REQUIRED_KEYS = {
    "bayesadapter_text_only_ckpt",
    "bayesadapter_text_only_mu_strategy",
    "bayesadapter_text_only_sigma_strategy",
}

LEGACY_BAYESADAPTER_TEXT_ONLY_KEYS = {
    "text_only_method_name",
    "bayesadapter_prior_source",
    "bayesadapter_prior_mu_mode",
    "bayesadapter_prior_mu_lambda",
    "bayesadapter_use_text_only_prior_sigma",
}
# ============================================================
# 3) 数据结构
# ============================================================

@dataclass
class ExperimentPlan:
    plan_path: Path
    base_config: dict[str, Any]
    experiments: list[dict[str, Any]]


# ============================================================
# 4) XML 基础解析
# ============================================================

def _read_text(elem: ET.Element) -> str:
    if elem.text is None:
        return ""
    return elem.text.strip()


def _parse_simple_element(elem: ET.Element) -> Any:
    children = list(elem)

    if not children:
        return _read_text(elem)

    # <xxx><item>...</item><item>...</item></xxx>
    if all(child.tag == "item" for child in children):
        return [_parse_simple_element(child) for child in children]

    # 当前实现要求 config 片段扁平；这里只保留解析能力，
    # 后面 coerce_flat_config 会拒绝 dict 值
    result: dict[str, Any] = {}
    for child in children:
        if child.tag in result:
            raise ValueError(
                f"XML 中出现重复标签 <{child.tag}>，当前实现要求配置键唯一。"
            )
        result[child.tag] = _parse_simple_element(child)
    return result


def _resolve_import_path(current_file: Path, href: str) -> Path:
    return (current_file.parent / href).resolve()


def _merge_layers(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    merged.update(override)
    return merged


# ============================================================
# 5) 读取 config 片段
# ============================================================

def _load_config_fragment_recursive(path: Path, active_stack: tuple[Path, ...]) -> dict[str, Any]:
    path = path.resolve()

    if path in active_stack:
        cycle = " -> ".join(str(p) for p in [*active_stack, path])
        raise ValueError(f"检测到循环 import: {cycle}")

    root = ET.parse(path).getroot()
    if root.tag != "config":
        raise ValueError(f"{path} 的根标签必须是 <config>，实际为 <{root.tag}>")

    merged: dict[str, Any] = {}

    imports_elem = root.find("imports")
    if imports_elem is not None:
        for import_elem in imports_elem.findall("import"):
            href = import_elem.attrib.get("href", "").strip()
            if not href:
                raise ValueError(f"{path} 中存在没有 href 的 <import>")
            child_path = _resolve_import_path(path, href)
            child_cfg = _load_config_fragment_recursive(child_path, (*active_stack, path))
            merged = _merge_layers(merged, child_cfg)

    for child in root:
        if child.tag == "imports":
            continue
        merged[child.tag] = _parse_simple_element(child)

    return merged


def load_config_fragment(path: str | Path) -> dict[str, Any]:
    return _load_config_fragment_recursive(Path(path), tuple())


# ============================================================
# 6) 读取 experiments 片段
# ============================================================

def _parse_experiment_element(elem: ET.Element) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for child in elem:
        if child.tag in result:
            raise ValueError(f"<experiment> 中重复键: {child.tag}")
        result[child.tag] = _parse_simple_element(child)
    return result


def _load_experiments_fragment(path: Path) -> list[dict[str, Any]]:
    path = path.resolve()
    root = ET.parse(path).getroot()

    if root.tag != "experiments":
        raise ValueError(f"{path} 的根标签必须是 <experiments>，实际为 <{root.tag}>")

    experiments: list[dict[str, Any]] = []
    for child in root:
        if child.tag != "experiment":
            raise ValueError(f"{path} 的 <experiments> 下只允许 <experiment>，实际为 <{child.tag}>")
        experiments.append(_parse_experiment_element(child))

    if not experiments:
        raise ValueError(f"{path} 中没有任何 <experiment>")

    return experiments


# ============================================================
# 7) 读取总 plan
# ============================================================

def load_experiment_plan(path: str | Path) -> ExperimentPlan:
    plan_path = Path(path).resolve()
    root = ET.parse(plan_path).getroot()

    if root.tag not in {"experiment_plan", "plan"}:
        raise ValueError(
            f"{plan_path} 的根标签必须是 <experiment_plan> 或 <plan>，实际为 <{root.tag}>"
        )

    base_config: dict[str, Any] = {}
    experiments: list[dict[str, Any]] = []

    imports_elem = root.find("imports")
    if imports_elem is not None:
        for import_elem in imports_elem.findall("import"):
            href = import_elem.attrib.get("href", "").strip()
            if not href:
                raise ValueError(f"{plan_path} 中存在没有 href 的 <import>")
            cfg_path = _resolve_import_path(plan_path, href)
            fragment = load_config_fragment(cfg_path)
            base_config = _merge_layers(base_config, fragment)

    for child in root:
        if child.tag in {"imports", "experiments"}:
            continue
        base_config[child.tag] = _parse_simple_element(child)

    experiments_elem = root.find("experiments")
    if experiments_elem is None:
        raise ValueError(f"{plan_path} 缺少 <experiments>")

    for child in experiments_elem:
        if child.tag == "experiment":
            experiments.append(_parse_experiment_element(child))
        elif child.tag == "import":
            href = child.attrib.get("href", "").strip()
            if not href:
                raise ValueError(f"{plan_path} 的 <experiments> 下存在没有 href 的 <import>")
            exp_path = _resolve_import_path(plan_path, href)
            experiments.extend(_load_experiments_fragment(exp_path))
        else:
            raise ValueError(
                f"{plan_path} 的 <experiments> 下只允许 <experiment> 或 <import>，实际为 <{child.tag}>"
            )

    if not experiments:
        raise ValueError(f"{plan_path} 中没有任何实验条目")

    return ExperimentPlan(
        plan_path=plan_path,
        base_config=base_config,
        experiments=experiments,
    )


# ============================================================
# 8) 类型转换
# ============================================================

def _to_bool(key: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False

    raise ValueError(f"{key} 必须是 true/false，当前值为: {value}")


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
            raise ValueError(f"{key} 必须写成 <item> 列表")
        return [int(str(x).strip()) for x in value]

    if isinstance(value, list):
        return [str(x) for x in value]

    if isinstance(value, dict):
        raise ValueError(
            f"当前配置实现要求所有 config 片段是扁平键值，键 {key} 解析成了嵌套对象，请改成扁平配置。"
        )

    return _coerce_scalar(key, value)


def coerce_flat_config(raw_cfg: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in raw_cfg.items():
        result[key] = _coerce_value(key, value)
    return result


# ============================================================
# 9) 规范化 / 校验
# ============================================================

def normalize_recipe_name(recipe_name: str) -> str:
    key = str(recipe_name).strip().lower()
    if key in {"deterministic_coop_standard", "deterministic_coop"}:
        return "deterministic_coop"
    if key in {"text_only_bayes_coop", "vlm_adapter"}:
        return key
    raise ValueError(
        f"未知 recipe_name: {recipe_name}，可选值为 "
        f"['text_only_bayes_coop', 'vlm_adapter', 'deterministic_coop']"
    )


def _require_keys(cfg: dict[str, Any], required: set[str], where: str) -> None:
    missing = sorted(k for k in required if k not in cfg)
    if missing:
        raise ValueError(f"{where} 缺少必须字段: {missing}")


def _validate_value_ranges(cfg: dict[str, Any]) -> None:
    if int(cfg["epochs"]) < 0:
        raise ValueError("epochs 必须 >= 0")
    if int(cfg["batch_size"]) <= 0:
        raise ValueError("batch_size 必须 > 0")
    if int(cfg["num_workers"]) < 0:
        raise ValueError("num_workers 必须 >= 0")
    if int(cfg["prediction_topk"]) <= 0:
        raise ValueError("prediction_topk 必须 > 0")
    if int(cfg["warmup_epoch"]) < 0:
        raise ValueError("warmup_epoch 必须 >= 0")
    if int(cfg["train_aug_repeats"]) < 1:
        raise ValueError("train_aug_repeats 必须 >= 1")
    if int(cfg["shots_per_class"]) < 0:
        raise ValueError("shots_per_class 必须 >= 0")
    if float(cfg["lr"]) <= 0:
        raise ValueError("lr 必须 > 0")
    if float(cfg["warmup_cons_lr"]) < 0:
        raise ValueError("warmup_cons_lr 必须 >= 0")
    if float(cfg["weight_decay"]) < 0:
        raise ValueError("weight_decay 必须 >= 0")

    if "bayesadapter_text_only_mu_blend_lambda" in cfg:
        lam = float(cfg["bayesadapter_text_only_mu_blend_lambda"])
        if not (0.0 <= lam <= 1.0):
            raise ValueError("bayesadapter_text_only_mu_blend_lambda 必须在 [0, 1] 内")



def _validate_enums(cfg: dict[str, Any]) -> None:
    optimizer = str(cfg["optimizer"]).strip().lower()
    if optimizer not in {"sgd", "adam", "adamw"}:
        raise ValueError("optimizer 必须是 ['sgd', 'adam', 'adamw'] 之一")

    scheduler = str(cfg["lr_scheduler"]).strip().lower()
    if scheduler not in {"none", "cosine"}:
        raise ValueError("lr_scheduler 必须是 ['none', 'cosine'] 之一")

    model_selection = str(cfg["model_selection"]).strip().lower()
    if model_selection not in {"best", "last"}:
        raise ValueError("model_selection 必须是 ['best', 'last'] 之一")

    selection_metric = str(cfg["selection_metric"]).strip().lower()
    if selection_metric not in {"loss", "acc", "nlpd", "ece"}:
        raise ValueError("selection_metric 必须是 ['loss', 'acc', 'nlpd', 'ece'] 之一")

    selection_mode = str(cfg["selection_mode"]).strip().lower()
    if selection_mode not in {"auto", "min", "max"}:
        raise ValueError("selection_mode 必须是 ['auto', 'min', 'max'] 之一")

    if "train_objective" in cfg:
        train_objective = str(cfg["train_objective"]).strip().lower()
        if train_objective not in {"map", "bayes", "hybrid"}:
            raise ValueError("train_objective 必须是 ['map', 'bayes', 'hybrid'] 之一")

    if "bayesadapter_covariance_mode" in cfg:
        cov_mode = str(cfg["bayesadapter_covariance_mode"]).strip().lower()
        if cov_mode not in {"paper_scalar", "diag"}:
            raise ValueError("bayesadapter_covariance_mode 必须是 ['paper_scalar', 'diag'] 之一")

    if "bayesadapter_text_only_mu_strategy" in cfg:
        mu_strategy = str(cfg["bayesadapter_text_only_mu_strategy"]).strip().lower()
        if mu_strategy not in {"replace", "blend"}:
            raise ValueError(
                "bayesadapter_text_only_mu_strategy 必须是 ['replace', 'blend'] 之一"
            )

    if "bayesadapter_text_only_sigma_strategy" in cfg:
        sigma_strategy = str(cfg["bayesadapter_text_only_sigma_strategy"]).strip().lower()
        if sigma_strategy not in {"ignore", "override"}:
            raise ValueError(
                "bayesadapter_text_only_sigma_strategy 必须是 ['ignore', 'override'] 之一"
            )


def validate_final_config(cfg: dict[str, Any]) -> None:
    _reject_legacy_bayesadapter_text_only_keys(cfg)

    _require_keys(cfg, REQUIRED_COMMON_KEYS, "公共配置")
    _require_keys(cfg, REQUIRED_EXPERIMENT_KEYS, "实验条目")

    recipe_name = normalize_recipe_name(str(cfg.get("recipe_name", "")).strip())
    recipe_required = RECIPE_REQUIRED_KEYS[recipe_name]
    _require_keys(cfg, recipe_required, f"recipe={recipe_name}")

    if recipe_name == "vlm_adapter":
        adapter_key = str(cfg["adapter_name"]).upper()

        if adapter_key in {"TR", "TASKRESIDUAL"}:
            _require_keys(cfg, TR_REQUIRED_KEYS, "TR 配置")

        elif adapter_key in {"CLIPA", "CLIPADAPTER"}:
            _require_keys(cfg, CLIPA_REQUIRED_KEYS, "CLIPA 配置")

        elif adapter_key in {"TIPA", "TIPADAPTER"}:
            _require_keys(cfg, TIPA_REQUIRED_KEYS, "TIPA 配置")

        elif adapter_key == "GAUSSIAN_PER_CLASS":
            _require_keys(cfg, GAUSSIAN_PER_CLASS_REQUIRED_KEYS, "GAUSSIAN_PER_CLASS 配置")

        elif adapter_key in {"BAYESADAPTER", "BAYES_ADAPTER"}:
            _require_keys(cfg, BAYESADAPTER_REQUIRED_KEYS, "BayesAdapter 配置")

            if _has_text_only_bridge(cfg):
                _require_keys(cfg, TEXT_ONLY_BRIDGE_REQUIRED_KEYS, "BayesAdapter text-only bridge 配置")

                if "bayesadapter_text_only_run_dir" not in cfg:
                    raise ValueError(
                        "当启用 bayesadapter_text_only bridge 时，"
                        "必须先根据 template 派生出 bayesadapter_text_only_run_dir。"
                    )

                mu_strategy = str(cfg["bayesadapter_text_only_mu_strategy"]).strip().lower()
                if mu_strategy == "blend" and "bayesadapter_text_only_mu_blend_lambda" not in cfg:
                    raise ValueError(
                        "当 bayesadapter_text_only_mu_strategy=blend 时，"
                        "必须提供 bayesadapter_text_only_mu_blend_lambda。"
                    )

    _validate_enums(cfg)
    _validate_value_ranges(cfg)


# ============================================================
# 10) 派生字段
# ============================================================



def materialize_derived_fields(cfg: dict[str, Any]) -> dict[str, Any]:
    result = dict(cfg)

    # 规范化 recipe_name
    result["recipe_name"] = normalize_recipe_name(str(result["recipe_name"]))

    # 显式写出 cache_image_features，避免依赖 train_experiment._ensure_common_flags 的兜底
    result["cache_image_features"] = not bool(result["disable_cache_image_features"])

    recipe_name = str(result["recipe_name"])
    adapter_key = str(result.get("adapter_name", "")).upper()

    if (
        recipe_name == "vlm_adapter"
        and adapter_key in {"BAYESADAPTER", "BAYES_ADAPTER"}
        and "bayesadapter_text_only_run_dir" not in result
        and "bayesadapter_text_only_run_dir_template" in result
    ):
        template = str(result["bayesadapter_text_only_run_dir_template"])

        format_env = {
            "save_dir": result["save_dir"],
            "dataset": result["dataset"],
            "shots_per_class": result["shots_per_class"],
            "seed": result["seed"],
        }

        try:
            result["bayesadapter_text_only_run_dir"] = template.format(**format_env)
        except KeyError as exc:
            raise ValueError(
                f"bayesadapter_text_only_run_dir_template 中使用了未知占位符: {exc}"
            ) from exc

    return result



# ============================================================
# 11) 对外接口
# ============================================================

def build_resolved_run_dicts(plan_path: str | Path) -> list[dict[str, Any]]:
    plan = load_experiment_plan(plan_path)

    base_config = coerce_flat_config(plan.base_config)
    run_dicts: list[dict[str, Any]] = []

    for idx, experiment_raw in enumerate(plan.experiments, start=1):
        experiment_cfg = coerce_flat_config(experiment_raw)

        # 优先级：被 experiment 覆盖
        merged = _merge_layers(base_config, experiment_cfg)
        merged = materialize_derived_fields(merged)
        validate_final_config(merged)

        merged["_experiment_index"] = idx
        run_dicts.append(merged)

    return run_dicts


def build_resolved_run_namespaces(plan_path: str | Path) -> list[Namespace]:
    run_dicts = build_resolved_run_dicts(plan_path)
    return [Namespace(**cfg) for cfg in run_dicts]
from __future__ import annotations

from typing import Any


def flatten_metrics_history(metrics_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    把按 epoch 存储的嵌套 metrics 展平成 CSV 友好的行结构。
    约定:
    - train / val / test 是 dict
    - 其他顶层标量字段直接平铺
    """
    rows: list[dict[str, Any]] = []

    for item in metrics_history:
        row: dict[str, Any] = {}

        for key, value in item.items():
            if key in {"train", "val", "test"} and isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    row[f"{key}_{sub_key}"] = sub_value
            else:
                row[key] = value

        rows.append(row)

    return rows
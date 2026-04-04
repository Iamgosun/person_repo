from __future__ import annotations

from typing import Any, Literal, Optional, Sequence, Type, Union

import torch
import torch.nn as nn

from bayesvlm.hessians import KroneckerFactorizedCovariance
from bayesvlm.text_priors import get_class_text_distributions
from bayesvlm.text_encoder import CLIPTextEncoder
from bayesvlm.image_encoder import CLIPImageEncoder
from bayesvlm.adapter import  LinearProbeAdapter


TextEncoderType = Union[CLIPTextEncoder]
TextStateKind = Literal["vector", "distribution"]



ADAPTER_REGISTRY = {
    "LP": LinearProbeAdapter,
}


class VLMAdapter(nn.Module):
    def __init__(
        self,
        cfg: Any,
        classnames: Sequence[str],
        text_covariance= None,
    ):
        super().__init__()
        self.cfg=cfg
        # 基模型路径/配置
        self.model_name_or_path=self.cfg.model_name_or_path
        # 适配器配置
        self.adapter_name=self.cfg.adapter_name
        self.initialization=self.cfg.initialization

        # 文本先验参数
        # 数据集名称用于模板的选择 dataset_name
        self.classnames = list(classnames)

        # 获取clip图像编码器（带投影层）
        self.image_encoder = CLIPImageEncoder.from_huggingface(model_name_or_path=self.model_name_or_path)
        # 获取clip文本编码器（带投影层）
        self.text_encoder = CLIPTextEncoder.from_huggingface(model_name_or_path=self.model_name_or_path)
        # 获取Clip缩放参数
        self.logit_scale = self.image_encoder.logit_scale  #模型参数nn

        # 获取协方差
        self.text_covariance = text_covariance
        # 获取类别文本先验分布
        self.class_text_distributions = get_class_text_distributions(
            dataset_name=cfg.datasetname,
            class_names=self.classnames,
            text_encoder=self.text_encoder,
            covariance=self.text_covariance
        )

        # 验证类别分布

        from .unfenxi import summarize_class_text_uncertainty,print_class_uncertainty_report
        # 验证类别分布代码
        self.class_uncertainty_summary = summarize_class_text_uncertainty(
            self.class_text_distributions,
            eps=1e-6,
            topk=10,
        )
        print_class_uncertainty_report(
            self.class_uncertainty_summary,
            topn=min(20, len(self.classnames)),
        )


        # # 获取适配器类型
        # self.adapter_cls =  ADAPTER_REGISTRY[self.adapter_name]
        # # 获取适配器文本先验类型
        # self.adapter_input_kind = self.adapter_cls.input_kind
        # self.adapter = self._build_adapter()



    def _build_adapter(self) :
        if self.adapter_input_kind == "vector":
            adapter_input = self.class_text_distributions.mean
        elif self.adapter_input_kind == "distribution":
            adapter_input = self.class_text_distributions
        else:
            raise ValueError(f"Unknown adapter input kind: {self.adapter_input_kind}")

        return self.adapter_cls( adapter_input,initialization="RANDOM")


    # 获取当前模型设备
    def _runtime_device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except Exception:
            return torch.device("cpu")


    def _image_encoder_dtype(self) -> torch.dtype:

        return next(self.image_encoder.parameters()).dtype

    def _encode_image(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(
            device=self._runtime_device(),
            dtype=self._image_encoder_dtype(),
        )
        image_features = self.image_encoder(image)
        # 在深度学习库（如 Hugging Face Transformers 或某些 CLIP 实现）中，模型的前向传播往往不仅返回主要特征，还会返回一些辅助信息。
        if isinstance(image_features, tuple):
            image_features = image_features[0]

        return image_features


    def forward(
        self,
        image: torch.Tensor,
        return_features: bool = False,
    ):
        image_features = self._encode_image(image)
        logits = self.adapter(image_features, self.logit_scale)

        if return_features:
            return logits, image_features

        return logits


    def forward_features(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        features = features.to(
            device=self._runtime_device(),
            dtype=self._image_encoder_dtype(),
        )
        logits = self.adapter(features, self.logit_scale)
        return logits
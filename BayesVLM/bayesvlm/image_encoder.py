from __future__ import annotations

from typing import Optional

import torch
from transformers import (
    CLIPVisionModelWithProjection,
    SiglipVisionModel,
)

from bayesvlm.common import EncoderResult, get_projection_dim, torch_load_cpu


def _extract_images(batch_or_images):
    """
    中文说明：
    兼容两种输入：
    1）batch["image"]
    2）直接传 tensor 图像
    """
    if isinstance(batch_or_images, dict):
        return batch_or_images["image"]
    return batch_or_images


class CLIPImageEncoder(torch.nn.Module):
    """
    中文说明：
    1. 从 vlm.py 中解耦出来的 CLIP 图像编码器。
    2. 预留了两类 adapter 扩展点：
       - input_adapter：作用在原始图像输入上
       - activation_adapter：作用在 pooled visual feature 上
    3. 后续你做图像侧 CoOp / adapter / visual prompt 时，可以直接接这里。
    """

    def __init__(
        self,
        vision_model: CLIPVisionModelWithProjection,
    ):
        super().__init__()
        self.vision_encoder = vision_model.vision_model
        self.vision_projection = vision_model.visual_projection
        self.device = getattr(vision_model, "device", None)

        # 中文说明：
        # 预留两个适配器入口。
        self.input_adapter = None
        self.activation_adapter = None

    def _get_device(self) -> torch.device:
        return next(self.parameters()).device

    @classmethod
    def from_huggingface(
        cls,
        model_name: str,
        projection_dim: Optional[int] = None,
        device: Optional[str] = None,
        local_files_only: bool = False,
    ):
        if projection_dim is None:
            projection_dim = get_projection_dim(
                model_name,
                local_files_only=local_files_only,
            )

        vision_model = CLIPVisionModelWithProjection.from_pretrained(
            model_name,
            projection_dim=projection_dim,
            local_files_only=local_files_only,
        )
        model = cls(vision_model)
        model = model.to(device) if device is not None else model
        model.device = device
        return model

    def save_projection_weights(self, path: str):
        torch.save(self.vision_projection.state_dict(), path)

    def load_projection_weights(
        self,
        *,
        path: Optional[str] = None,
        state_dict: Optional[dict] = None,
    ):
        if state_dict is not None:
            self.vision_projection.load_state_dict(state_dict)
            return

        if path is None:
            raise ValueError("Either path or state_dict must be provided.")

        self.vision_projection.load_state_dict(torch_load_cpu(path))

    def freeze_all_layers(self):
        for param in self.parameters():
            param.requires_grad = False

    def freeze_backbone(self):
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

    def freeze_all_layers_except_projection(self):
        self.freeze_all_layers()
        for param in self.vision_projection.parameters():
            param.requires_grad = True

    # 兼容旧代码里的拼写
    def freeze_all_layers_exept_projection(self):
        self.freeze_all_layers_except_projection()

    def enable_gradients(
        self,
        k_last_layers: int = 0,
        enable_projection: bool = True,
    ):
        if enable_projection:
            self.vision_projection.train()
            for param in self.vision_projection.parameters():
                param.requires_grad = True

        if k_last_layers > 0:
            for layer in self.vision_encoder.encoder.layers[-k_last_layers:]:
                layer.train()
                for param in layer.parameters():
                    param.requires_grad = True

    def register_input_adapter(self, adapter: torch.nn.Module):
        """
        中文说明：
        给原始图像输入注册 adapter。
        """
        self.input_adapter = adapter

    def register_activation_adapter(self, adapter: torch.nn.Module):
        """
        中文说明：
        给 pooled visual feature 注册 adapter。
        """
        self.activation_adapter = adapter

    def clear_adapters(self):
        self.input_adapter = None
        self.activation_adapter = None

    def forward_features(self, batch_or_images):
        images = _extract_images(batch_or_images).to(self._get_device())

        if self.input_adapter is not None:
            images = self.input_adapter(images)

        image_input = dict(pixel_values=images)
        image_outputs = self.vision_encoder(**image_input)
        activations = image_outputs[1]

        if self.activation_adapter is not None:
            activations = self.activation_adapter(activations)

        return activations

    def project_features(self, activations: torch.Tensor):
        return self.vision_projection(activations)

    def forward_from_activations(
        self,
        activations: torch.Tensor,
        return_activations: bool = False,
    ):
        image_embeds = self.project_features(activations)

        if return_activations:
            return EncoderResult(
                embeds=image_embeds,
                activations=activations,
            )

        return image_embeds

    def forward(self, batch, return_activations=False):
        activations = self.forward_features(batch)
        return self.forward_from_activations(
            activations=activations,
            return_activations=return_activations,
        )


class SiglipVisionEncoderWithoutProjection(torch.nn.Module):
    def __init__(
        self,
        model: SiglipVisionModel,
    ):
        super().__init__()
        self.vision_model = model.vision_model

    def forward(self, pixel_values: torch.Tensor):
        hidden_states = self.vision_model.embeddings(pixel_values)
        encoder_outputs = self.vision_model.encoder(inputs_embeds=hidden_states)

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.vision_model.post_layernorm(last_hidden_state)

        batch_size = last_hidden_state.shape[0]
        probe = self.vision_model.head.probe.repeat(batch_size, 1, 1)
        last_hidden_state = self.vision_model.head.attention(
            probe,
            last_hidden_state,
            last_hidden_state,
        )[0]

        residual = last_hidden_state
        last_hidden_state = self.vision_model.head.layernorm(last_hidden_state)
        mlp = self.vision_model.head.mlp

        last_hidden_state = mlp.fc1(last_hidden_state)
        last_hidden_state = mlp.activation_fn(last_hidden_state)

        return last_hidden_state, residual


class SiglipImageEncoder(torch.nn.Module):
    """
    中文说明：
    1. 从 vlm.py 中解耦出来的 SigLIP 图像编码器。
    2. 同样预留 input_adapter / activation_adapter 两种扩展点。
    3. 保留原来 residual + projection 的实现逻辑。
    """

    def __init__(
        self,
        vision_model: SiglipVisionModel,
    ):
        super().__init__()
        self.vision_encoder = SiglipVisionEncoderWithoutProjection(vision_model)
        self.vision_projection = vision_model.vision_model.head.mlp.fc2
        self.device = getattr(vision_model, "device", None)
        self._raw_vision_model = vision_model

        self.input_adapter = None
        self.activation_adapter = None

    def _get_device(self) -> torch.device:
        return next(self.parameters()).device

    @classmethod
    def from_huggingface(
        cls,
        model_name: str,
        device: Optional[str] = None,
        local_files_only: bool = False,
    ):
        vision_model = SiglipVisionModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        model = cls(vision_model)
        model = model.to(device) if device is not None else model
        model.device = device
        return model

    def save_projection_weights(self, path: str):
        torch.save(self.vision_projection.state_dict(), path)

    def load_projection_weights(
        self,
        *,
        path: Optional[str] = None,
        state_dict: Optional[dict] = None,
    ):
        if state_dict is not None:
            self.vision_projection.load_state_dict(state_dict)
            return

        if path is None:
            raise ValueError("Either path or state_dict must be provided.")

        self.vision_projection.load_state_dict(torch_load_cpu(path))

    def freeze_all_layers(self):
        for param in self.parameters():
            param.requires_grad = False

    def freeze_backbone(self):
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

    def freeze_all_layers_except_projection(self):
        self.freeze_all_layers()
        for param in self.vision_projection.parameters():
            param.requires_grad = True

    # 兼容旧代码里的拼写
    def freeze_all_layers_exept_projection(self):
        self.freeze_all_layers_except_projection()

    def enable_gradients(
        self,
        k_last_layers: int = 0,
        enable_projection: bool = True,
    ):
        if enable_projection:
            self.vision_projection.train()
            for param in self.vision_projection.parameters():
                param.requires_grad = True

        if k_last_layers > 0:
            # 中文说明：
            # 这里沿用你当前仓库里修正后的访问路径。
            for layer in self._raw_vision_model.vision_model.encoder.layers[-k_last_layers:]:
                layer.train()
                for param in layer.parameters():
                    param.requires_grad = True

    def register_input_adapter(self, adapter: torch.nn.Module):
        self.input_adapter = adapter

    def register_activation_adapter(self, adapter: torch.nn.Module):
        self.activation_adapter = adapter

    def clear_adapters(self):
        self.input_adapter = None
        self.activation_adapter = None

    def forward_features(self, batch_or_images):
        images = _extract_images(batch_or_images).to(self._get_device())

        if self.input_adapter is not None:
            images = self.input_adapter(images)

        activations, residuals = self.vision_encoder(images)

        activations = activations[:, 0]
        residuals = residuals[:, 0]

        if self.activation_adapter is not None:
            activations = self.activation_adapter(activations)

        return activations, residuals

    def project_features(
        self,
        activations: torch.Tensor,
        residuals: Optional[torch.Tensor] = None,
    ):
        if residuals is None:
            residuals = 0.0
        return self.vision_projection(activations) + residuals

    def forward_from_activations(
        self,
        activations: torch.Tensor,
        residuals: Optional[torch.Tensor] = None,
        return_activations: bool = False,
    ):
        image_embeds = self.project_features(
            activations=activations,
            residuals=residuals,
        )

        if return_activations:
            if residuals is None:
                residuals = torch.zeros_like(image_embeds)
            return EncoderResult(
                embeds=image_embeds,
                activations=activations,
                residuals=residuals,
            )

        return image_embeds

    def forward(self, batch, return_activations=False):
        activations, residuals = self.forward_features(batch)
        return self.forward_from_activations(
            activations=activations,
            residuals=residuals,
            return_activations=return_activations,
        )
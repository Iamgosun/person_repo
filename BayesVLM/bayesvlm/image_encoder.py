from typing import Optional, Sequence

import torch

from transformers import CLIPModel

class CLIPImageEncoder(torch.nn.Module):
    def __init__(self, clip_model: CLIPModel):
        super().__init__()

        # 从完整 CLIPModel 中提取视觉侧组件
        self.projection_dim = clip_model.config.projection_dim
        self.vision_encoder = clip_model.vision_model
        self.vision_projection = clip_model.visual_projection
        # logit_scale与图像编码器无关，只是为了方便获取
        self.logit_scale = clip_model.logit_scale  # nn.Parameter
        self.logit_scale.requires_grad = False

    @classmethod
    def from_huggingface(
        cls,
        model_name_or_path: str,
        local_files_only: bool = False,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Sequence[str] = ("q_proj", "v_proj"),
        train_projection: bool = False,
    ):
        clip_model = CLIPModel.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        )

        model = cls(clip_model)

        if use_lora:
            model.add_lora(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                train_projection=train_projection,
            )
        # 不添加lora则冻结整个模型
        else:
            model.freeze_all_layers()

        return model

    def freeze_all_layers(self):
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        for param in self.vision_projection.parameters():
            param.requires_grad = False

    def add_lora(
        self,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Sequence[str] = ("q_proj", "v_proj"),
        train_projection: bool = True
    ):
        """
        给视觉编码器加 LoRA。
        默认只给 q_proj / v_proj 加，这是最常见也最稳妥的做法之一。
        """
        from peft import LoraConfig, get_peft_model

        self.freeze_all_layers()

        peft_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=list(target_modules),
            lora_dropout=lora_dropout,
            bias="none",
        )

        self.vision_encoder = get_peft_model(self.vision_encoder, peft_config)

        if train_projection:
            self.vision_projection.train()
            for param in self.vision_projection.parameters():
                param.requires_grad = True
        return self


    def forward(
        self,
        batch,
        return_activations: bool = False
    ):

        device = next(self.parameters()).device
        dtype = self.vision_projection.weight.dtype
        images = batch["image"].to(device=device, dtype=dtype)

        vision_outputs = self.vision_encoder(
            pixel_values=images,
            return_dict=True,
        )

        image_pooled_output = vision_outputs.pooler_output
        image_embeds = self.vision_projection(image_pooled_output)

        if return_activations:
            return image_embeds, image_pooled_output

        return image_embeds
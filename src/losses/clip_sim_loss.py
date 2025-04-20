from typing import Optional, Literal
import torch
from torch import nn
from transformers import CLIPModel


class CLIPSimilarityLoss(torch.nn.Module):
    def __init__(
        self,
        sim_type: Literal["img2img", "txt2img"],
        clip_model: Optional[CLIPModel] = None,
        model_name="openai/clip-vit-base-patch16",
    ):
        super(CLIPSimilarityLoss, self).__init__()

        if sim_type not in ["img2img", "txt2img"]:
            raise ValueError(
                f"sim_type should be either 'img2img' or 'txt2img', but got {sim_type}"
            )
        self.sim_type = sim_type

        self.clip_model = clip_model or CLIPModel.from_pretrained(model_name)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, attention_mask: torch.Tensor = None
    ):
        """
        Compute the CLIP similarity loss between two tensors.
        Args:
            x1: First tensor (image or text features).
            x2: Second tensor (image features).
            attention_mask: Attention mask for text features.
        """
        B = x1.shape[0]
        # Get the text and image features
        with torch.no_grad():
            if self.sim_type == "img2img":
                x_features = self.clip_model.get_image_features(
                    pixel_values=torch.cat([x1, x2], dim=0)
                )
                x1_features = x_features[:B]
                x2_features = x_features[B:]
            elif self.sim_type == "txt2img":
                x1_features = self.clip_model.get_text_features(
                    input_ids=x1, attention_mask=attention_mask
                )
                x2_features = self.clip_model.get_image_features(pixel_values=x2)
            else:
                raise ValueError(
                    f"sim_type should be either 'img2img' or 'txt2img', but got {self.sim_type}"
                )
        # Compute cosine similarity
        similarity = self.cosine_similarity(x1_features, x2_features)

        # Compute the loss
        loss = 1 - similarity.mean()

        return loss

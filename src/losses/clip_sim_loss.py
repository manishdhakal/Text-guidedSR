from typing import Optional
import torch
from torch import nn
from transformers import CLIPModel

class CLIPSimilarityLoss(torch.nn.Module):
    def __init__(self, clip_model: Optional[CLIPModel] = None, model_name='openai/clip-vit-base-patch16'):
        super(CLIPSimilarityLoss, self).__init__()
        
        self.clip_model = clip_model or CLIPModel.from_pretrained(model_name)
        
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
    
    def forward(self, input_ids: torch.Tensor, pixel_values: torch.Tensor, attention_mask: torch.Tensor = None):
        # Get the text and image features
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
            image_features = self.clip_model.get_image_features(pixel_values=pixel_values)

        # Normalize the features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity
        similarity = self.cosine_similarity(text_features, image_features)

        # Compute the loss
        loss = 1 - similarity.mean()
        
        return loss
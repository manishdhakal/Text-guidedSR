import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from transformers import CLIPModel
from diffusers import UNet2DModel



class ConditionalUNet(nn.Module):
    def __init__(
        self,
        model_name="openai/clip-vit-base-patch16",
        hr_size=224,
        lr_emb_size=128,
        text_emb_size=128,
        layers_per_block=2,
        block_out_channels=(32, 64, 128),
        scale=8,
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "AttnUpBlock2D",
        ),
        *args,
        **kwargs,
    ):
        super(ConditionalUNet, self).__init__(*args, **kwargs)

        self.text_emb_size = text_emb_size
        self.hr_size = hr_size
        self.scale = scale
        
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.proj_lr = nn.Linear(self.clip_model.config.projection_dim, lr_emb_size)

        if text_emb_size > 0:
            self.proj_text = nn.Linear(
                self.clip_model.config.projection_dim, text_emb_size
            )

        cond_emb_size = lr_emb_size + text_emb_size
        self.model = UNet2DModel(
            sample_size=hr_size,
            in_channels=3 + cond_emb_size,
            out_channels=3,
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
        )


    def forward(
        self,
        x: torch.Tensor,
        t: torch.LongTensor,
        lr_image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ):

        lr_image = TF.resize(
            lr_image,
            (self.hr_size, self.hr_size),
            interpolation=TF.InterpolationMode.BICUBIC,
        )
        # Get the text and image features
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(pixel_values=lr_image)
            lr_emb = self.proj_lr(image_features)

            if self.text_emb_size > 0:
                text_features = self.clip_model.get_text_features(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                text_emb = self.proj_text(text_features)
            else:
                text_emb = None

        # Expand the features to match H and W
        lr_emb = (
            lr_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], x.shape[3])
        )
        cond_emb = lr_emb
        if text_emb is not None:
            text_emb = (
                text_emb.unsqueeze(-1)
                .unsqueeze(-1)
                .expand(-1, -1, x.shape[2], x.shape[3])
            )
            cond_emb = torch.cat([lr_emb, text_emb], dim=1)

        # Concatenate the conditioning features with the input
        x = torch.cat([x, cond_emb], dim=1)

        # Pass through the UNet model
        return self.model(x, t).sample


if __name__ == "__main__":
    # Example usage
    model = ConditionalUNet()
    x = torch.randn(1, 3, 224, 224)
    t = torch.randint(0, 1000, (1,))
    lr_image = torch.randn(1, 3, 224, 224)
    input_ids = torch.randint(0, 1000, (1, 77))
    attention_mask = torch.ones((1, 77))

    output = model(x, t, lr_image, input_ids, attention_mask)
    print(output.shape)  # Should be (1, 3, 224, 224)

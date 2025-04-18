from typing import Optional, Tuple, List
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from transformers import CLIPModel, CLIPSegConfig, ViTModel
from transformers.models.clipseg.modeling_clipseg import (
    CLIPSegDecoderLayer,
    CLIPSegDecoderOutput,
)

from torchvision.transforms import functional as TF

torch.autograd.set_detect_anomaly(True)

# Define a UNet Model
class UNet(nn.Module):
    def __init__(self, in_channels=3, hid_channels=[64, 128, 256, 512], out_channels=3,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels
        
        self.exciter = nn.Conv2d(in_channels, hid_channels[0], kernel_size=3, padding=1)
        self.squeezer = nn.Conv2d(hid_channels[0] * 2, out_channels, kernel_size=3, padding=1)
        
        self.downsamplers = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder layers
        for i in range(len(hid_channels)-1):
            in_ch = hid_channels[i]
            out_ch = hid_channels[i+1]
            self.downsamplers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.ReLU(),
                    # nn.Conv2d(hid_channels[i+1], hid_channels[i+1], kernel_size=3, padding=1),
                    # nn.ReLU()
                )
            )
        
        # Decoder layers
        for i in range(len(hid_channels)-1, 0, -1):
            if i == len(hid_channels)-1:
                in_ch = hid_channels[i]
            else:
                in_ch = hid_channels[i] * 2  
            out_ch = hid_channels[i-1]
    
            self.upsamplers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                    nn.ReLU(),
                    # nn.Conv2d(hid_channels[i-1], hid_channels[i-1], kernel_size=3, padding=1),
                    # nn.ReLU()
                )
            )

    def forward(self, x):
        
        # Downsampling
        x = self.exciter(x)
        skip_connections = []
        for downsampler in self.downsamplers:
            skip_connections.append(x)
            x = downsampler(x)
            x = self.pool(x)
        
        # Bottleneck
        x = F.relu(x)
        
        # Upsampling
        for i, upsampler in enumerate(self.upsamplers):
            x = upsampler(x)
            x = torch.cat((x, skip_connections[-(i+1)]), dim=1)
        
        x = self.squeezer(x)
        
        return x


class CondUNet(nn.Module):
    def __init__(self, in_channels=3, hid_channels=[64, 128, 256, 512], out_channels=3,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels
        
        self.exciter = nn.Conv2d(in_channels, hid_channels[0], kernel_size=3, padding=1)
        self.squeezer = nn.Conv2d(hid_channels[0] * 2, out_channels, kernel_size=3, padding=1)
        
        self.downsamplers = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder layers
        for i in range(len(hid_channels)-1):
            in_ch = hid_channels[i]
            out_ch = hid_channels[i+1]
            self.downsamplers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.ReLU(),
                    # nn.Conv2d(hid_channels[i+1], hid_channels[i+1], kernel_size=3, padding=1),
                    # nn.ReLU()
                )
            )
        
        # Decoder layers
        for i in range(len(hid_channels)-1, 0, -1):
            if i == len(hid_channels)-1:
                in_ch = hid_channels[i]
            else:
                in_ch = hid_channels[i] * 2  
            out_ch = hid_channels[i-1]
    
            self.upsamplers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                    nn.ReLU(),
                    # nn.Conv2d(hid_channels[i-1], hid_channels[i-1], kernel_size=3, padding=1),
                    # nn.ReLU()
                )
            )

        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        
        
        for parameter in self.clip_encoder.parameters():
            parameter.requires_grad = False
        
        self.film_muls = nn.ModuleList()
        self.film_adds = nn.ModuleList() 
        
        clip_projection_dim = self.clip.config.projection_dim
        
        for h in hid_channels:
            self.film_muls.append(nn.Linear(clip_projection_dim, h))
            self.film_adds.append(nn.Linear(clip_projection_dim, h))
        
    def forward(self, x, lr_image):
        
        hr_h, hr_w = x.shape[2], x.shape[3]
        interpolated_lr = TF.resize(lr_image, (hr_h, hr_w), interpolation=TF.InterpolationMode.BICUBIC)    
        interpolated_lr = self.clip.get_image_features(interpolated_lr)

        # Downsampling
        x = self.exciter(x)
        skip_connections = []
        for i, downsampler in enumerate(self.downsamplers):
            
            # Apply FiLM for conditioning
            film_mul = self.film_muls[i](interpolated_lr)
            film_add = self.film_adds[i](interpolated_lr)
            x = film_mul * x.permute(1,2,0) + film_add
            x = x.permute(2,0,1)
            
            skip_connections.append(x)
            x = downsampler(x)
            x = self.pool(x)
        
        # Bottleneck
        x = F.relu(x)
        
        # Upsampling
        for i, upsampler in enumerate(self.upsamplers):
            x = upsampler(x)
            x = torch.cat((x, skip_connections[-(i+1)]), dim=1)
        
        x = self.squeezer(x)
        
        return x
    
def PatchUpsample(x, scale):
    n, c, h, w = x.shape
    x = torch.zeros(n, c, h, scale, w, scale).to(x.device) + x.view(n, c, h, 1, w, 1)
    return x.view(n, c, h * scale, w * scale)
    
class CLIPSegDecoder(nn.Module):
    def __init__(self, config: CLIPSegConfig):
        super().__init__()

        self.conditional_layer = config.conditional_layer

        self.film_mul = nn.Linear(config.projection_dim, config.reduce_dim)
        self.film_add = nn.Linear(config.projection_dim, config.reduce_dim)

        if config.use_complex_transposed_convolution:
            transposed_kernels = (
                config.vision_config.patch_size // 4,
                config.vision_config.patch_size // 4,
            )

            self.transposed_convolution = nn.Sequential(
                nn.Conv2d(
                    config.reduce_dim, config.reduce_dim, kernel_size=3, padding=1
                ),
                nn.ReLU(),
                nn.ConvTranspose2d(
                    config.reduce_dim,
                    config.reduce_dim // 2,
                    kernel_size=transposed_kernels[0],
                    stride=transposed_kernels[0],
                ),
                nn.ReLU(),
                nn.ConvTranspose2d(
                    config.reduce_dim // 2,
                    3,
                    kernel_size=transposed_kernels[1],
                    stride=transposed_kernels[1],
                ),
            )
        else:
            self.transposed_convolution = nn.ConvTranspose2d(
                config.reduce_dim,
                3,
                config.vision_config.patch_size,
                stride=config.vision_config.patch_size,
            )

        depth = len(config.extract_layers)
        self.reduces = nn.ModuleList(
            [
                nn.Linear(config.vision_config.hidden_size, config.reduce_dim)
                for _ in range(depth)
            ]
        )

        decoder_config = copy.deepcopy(config.vision_config)
        decoder_config.hidden_size = config.reduce_dim
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        decoder_config.hidden_act = "relu"
        self.layers = nn.ModuleList(
            [
                CLIPSegDecoderLayer(decoder_config)
                for _ in range(len(config.extract_layers))
            ]
        )

    def forward(
        self,
        hidden_states: Tuple[torch.Tensor],
        conditional_embeddings: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        activations = hidden_states[::-1]

        output = None
        for i, (activation, layer, reduce) in enumerate(
            zip(activations, self.layers, self.reduces)
        ):
            if output is not None:
                output = reduce(activation) + output
            else:
                output = reduce(activation)

            if i == self.conditional_layer:
                output = self.film_mul(conditional_embeddings) * output.permute(
                    1, 0, 2
                ) + self.film_add(conditional_embeddings)
                output = output.permute(1, 0, 2)

            layer_outputs = layer(
                output,
                attention_mask=None,
                causal_attention_mask=None,
                output_attentions=output_attentions,
            )

            output = layer_outputs[0]

            if output_hidden_states:
                all_hidden_states += (output,)

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        output = output[:, 1:, :].permute(
            0, 2, 1
        )  # remove cls token and reshape to [batch_size, reduce_dim, seq_len]

        size = int(math.sqrt(output.shape[2]))

        batch_size = conditional_embeddings.shape[0]
        output = output.view(batch_size, output.shape[1], size, size)

        logits = self.transposed_convolution(output).squeeze(1)

        if not return_dict:
            return tuple(
                v for v in [logits, all_hidden_states, all_attentions] if v is not None
            )

        return CLIPSegDecoderOutput(
            logits=logits,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

def get_beta_schedule(T, start=1e-4, end=0.02):
    return torch.linspace(start, end, T)


class CLIPSR(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        hr_size: int = 224,
        timesteps : int = 100,
    ) -> None:
        """Initialize a `SimpleDenseNet` module.

        :param input_size: The number of input features.
        :param lin1_size: The number of output features of the first linear layer.
        :param lin2_size: The number of output features of the second linear layer.
        :param lin3_size: The number of output features of the third linear layer.
        :param output_size: The number of output features of the final linear layer.
        """
        super().__init__()

        self.hr_size = hr_size

        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.config = CLIPSegConfig().from_pretrained("CIDAS/clipseg-rd64-refined")
        self.config.reduce_dim = 64
        # self.config.extract_layers = [0,1,2,3,4,5,6,7,8,9,10,11]
        self.sr_decoder = CLIPSegDecoder(config=self.config)

        self.timesteps = timesteps
        
        self.betas = get_beta_schedule(timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        for parameter in self.clip.parameters():
            parameter.requires_grad = False
    
    # def fwd_noise(lr_interpolated, hr_gt):
    #     noise = lr_interpolated - hr_gt
    #     alpha_cumpord_t = alpha_cumpord.view(-1, 1, 1, 1)
        
    def forward(
        self,
        image_lr: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        # timestep: int,
        **kwargs: Optional[dict],
    ) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """

        # alpha_cumpord_t = self.alphas_cumprod[timestep]
        lr_size = image_lr.shape[2]
        
        A = nn.AdaptiveAvgPool2d((lr_size, lr_size))
        Ap = lambda z: PatchUpsample(z, self.hr_size // lr_size)
        
        interpolated = TF.resize(
            image_lr, (self.hr_size, self.hr_size), interpolation=TF.InterpolationMode.BICUBIC
        )
        
        # for t in range(self.timesteps):
        with torch.no_grad():
            vision_outputs = self.clip.vision_model(
                pixel_values=interpolated,
                output_hidden_states=True,
            )
            pooled_output = self.clip.visual_projection(vision_outputs[1])

            hidden_states = vision_outputs.hidden_states
            # we add +1 here as the hidden states also include the initial embeddings
            activations = [hidden_states[i + 1] for i in self.config.extract_layers]

            # text_outputs = self.clip.text_model(
            #     input_ids=input_ids,
            #     attention_mask=attention_mask,
            # )

            # pooled_output = text_outputs[1]
            # text_features = self.clip.text_projection(pooled_output)
            text_features = self.clip.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            decoder_outputs = self.sr_decoder(
                activations,
                text_features,
            )
            # print(decoder_outputs[0])
            noise = decoder_outputs[0]
        
        return reconstructed_image


if __name__ == "__main__":
    # model = CLIPSR()
    # print(model.config)
    # image_lr = torch.randn(16, 3, 224, 224)
    # input_ids = torch.randint(0, 1000, (16, 77))
    # attention_mask = torch.ones((16, 77))
    # output = model(image_lr, input_ids, attention_mask)
    # output = torch.sum(output)
    # output.backward()
    
    unet = UNetModel()
    image = torch.randn(16, 3, 224, 224)
    output = unet(image)
    print(output.shape)

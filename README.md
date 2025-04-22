<div align="center">

# Text Guided Super Resolution

[![python](https://img.shields.io/badge/-Python_3.8_%7C_3.9_%7C-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/) <br>
[![tests](https://github.com/ashleve/lightning-hydra-template/actions/workflows/test.yml/badge.svg)](https://github.com/ashleve/lightning-hydra-template/actions/workflows/test.yml)
[![code-quality](https://github.com/ashleve/lightning-hydra-template/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/ashleve/lightning-hydra-template/actions/workflows/code-quality-main.yaml)<br>
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/ashleve/lightning-hydra-template/pulls)

Use Text as control signals to guide the segmentation tasks 🚀⚡🔥<br>

</div>

<br>

## 📌  Introduction
This a repo for the final project submission of one my course work. With this work, I am implementing a simple difusion model with UNet that uses text and low resolution (LR) image to generate a high resolution (HR) image.

## Installation
Initialize a virtual environment of `Python` (especially 3.9 version). Install all of the packages from the `requirements.txt` as:

```bash
pip install -r requirements.txt
```

## Dataset Structure
```kotlin
data/
├── images/
│   ├── train2017/
│   │   ├── 000000000009.jpg
│   │   ├── 000000000025.jpg
│   │   └── ...
│   ├── val2017/
│   │   ├── 000000000139.jpg
│   │   ├── 000000000285.jpg
│   │   └── ...
│   ├── llava1.5_coco2017train_captions.json
│   └── llava1.5_coco2017val_captions.json
```
The `images` directory contains only HR images.
If you want to use custom datasets, please update

### 📝 Caption Format
Each of the caption files (`llava1.5_coco2017train_captions.json` and `llava1.5_coco2017val_captions.json`) is structured as a simple dictionary:
```json
{
  "000000000009.jpg": "The image displays a variety of food items in different containers. There are two bowls, one containing broccoli and the other containing fruit. The broccoli is placed in the center of the bowl, while the fruit is spread out in the other bowl. Additionally, there are two sandwiches, one located on the left side of the image and the other on.",
  "000000000025.jpg": "The image features a giraffe standing next to a tree in a wooded area. The giraffe is positioned near the center of the scene, with its long neck and legs visible. The tree is located towards the left side of the image, providing a natural backdrop for the giraffe. The scene appears to be a peaceful, natural environment where the g",
  ...
}

```

## Training

```bash
for scale in 2 4 8 16;
do
  python src/train.py experiment=tsr${scale}x
done
```
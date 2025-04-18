from typing import Any, Dict, Optional, Tuple, Union
import json
import random
import os
from PIL import Image
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from transformers import CLIPTokenizer


class TSRDataset(Dataset):
    r"""
    Text-LR-HR Dataset
    Args:
        tokenizer_type (TOKENIZER_TYPE): Type of tokenizer to use
        prompt_types (List[PROMPT_TYPE]): List of prompt types to use
        images_dir (str): Path to images directory
        masks_dir (str): Path to masks directory
        caps_file (Optional[str], optional): Path to captions file. Defaults to None.
        img_size (int,int): Size of image. Defaults to (224, 224).
        context_length (int, optional): Context length. Defaults to 77.
        img_transforms (Optional[A.Compose], optional): Transforms to apply to image. Defaults to None.
        mask_transforms (Optional[A.Compose], optional): Transforms to apply to mask. Defaults to None.
        override_prompt (Optional[str], optional): Text uesd for overriding prompt. Defaults to None.
        zero_prompt (bool, optional): Whether to send zero in the place of prompt. Defaults to False.
        data_num (Optional[int | float], optional): Number of data to use. For float Defaults to 1.0.

    Raises:
        TypeError: If tokenizer_type is not one of TOKENIZER_TYPE
        ValueError: If data_num is of type float and is not in range [0., 1.]
    """

    def __init__(
        self,
        images_dir: str,
        caps_file: str,
        scale: int = 8,
        img_size: Tuple[int, int] = (224, 224),
        context_length: int = 77,
        img_transforms: Optional[T.Compose] = None,
        data_num: Union[int, float] = 1.0,
    ) -> None:
        super().__init__()

        self.img_size = img_size
        self.images_dir = images_dir
        self.img_files = os.listdir(images_dir)
        self.img_transforms = img_transforms
        self.context_length = context_length
        self.data_num = data_num
        self.scale = scale

        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")

        with open(caps_file, "r") as fp:
            self.imgs_captions = json.load(fp)

        random.shuffle(self.img_files)
        if type(self.data_num) == float:
            if self.data_num < 0 or self.data_num > 1:
                raise ValueError(
                    f"data_num must be in range [0, 1], OR must be +ve int but got {self.data_num} instead."
                )
            self.img_files = self.img_files[: int(len(self.img_files) * self.data_num)]
        else:
            self.img_files = self.img_files[: self.data_num]

        # Assign default img_transforms if no img_transforms is passed
        if self.img_transforms is None:
            self.img_transforms = T.Compose(
                [
                    T.Resize(size=img_size),
                    T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )

        # # Assign default mask_transforms if no mask_transforms is passed
        # if self.mask_transforms is None:
        #     self.mask_transforms = T.Compose(
        #         [
        #             T.Resize(
        #                 size=img_size,
        #                 interpolation=T.InterpolationMode.NEAREST_EXACT,
        #             ),
        #             T.ToTensor(),
        #         ]
        #     )

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index) -> Dict[str, Any]:
        img_f = self.img_files[index]

        # Ensure the image is read with RGB channels
        image = Image.open(f"{self.images_dir}/{img_f}").convert("RGB")
        image_hr = self.img_transforms(image)

        image_lr = TF.resize(
            image_hr,
            size=(self.img_size[0] // self.scale, self.img_size[1] // self.scale),
        )

        tokenizer_op = self.tokenizer(
            text=self.imgs_captions[img_f],
            max_length=self.context_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = tokenizer_op["input_ids"][0]
        attention_mask = tokenizer_op["attention_mask"][0]

        return dict(
            image_lr=image_lr,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_hr=image_hr,
        )


class TSRDataModule(LightningDataModule):
    """`LightningDataModule` for the TSR dataset.

    TSR is a dataset for text-guided image super-resolution.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (20_000, 5_000, 5_000),
        img_size: int = 224,
        scale: int = 8,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `MNISTDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    ([0.48145466, 0.4578275, 0.40821073]),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        # MNIST(self.hparams.data_dir, train=True, download=True)
        # MNIST(self.hparams.data_dir, train=False, download=True)
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = TSRDataset(
                images_dir=os.path.join(self.hparams.data_dir, "images/train2017"),
                caps_file=os.path.join(
                    self.hparams.data_dir, "llava1.5_coco2017train_captions.json"
                ),
                img_size=(self.hparams.img_size, self.hparams.img_size),
                scale=self.hparams.scale,
                context_length=77,
                data_num=self.hparams.train_val_test_split[0]
                + self.hparams.train_val_test_split[
                    1
                ],  # train and val split from the train set
            )

            self.data_train, self.data_val = random_split(
                dataset=trainset,
                lengths=self.hparams.train_val_test_split[:2],
                generator=torch.Generator(),
            )

            self.data_test = TSRDataset(
                images_dir=os.path.join(self.hparams.data_dir, "images/val2017"),
                caps_file=os.path.join(
                    self.hparams.data_dir, "llava1.5_coco2017val_captions.json"
                ),
                img_size=(self.hparams.img_size, self.hparams.img_size),
                scale=self.hparams.scale,
                context_length=77,
                data_num=self.hparams.train_val_test_split[2],
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    datamodule = TSRDataModule(
        data_dir="data/",
        train_val_test_split=(20_000, 5_000, 5_000),
        batch_size=64,
        num_workers=4,
        pin_memory=False,
    )
    # train_module = datamodule.train_dataloader()
    
    # train_point = next(iter(train_module))
    
    trainset = TSRDataset(
        images_dir=os.path.join("data/images/train2017"),
        caps_file=os.path.join("data/llava1.5_coco2017train_captions.json"),
        img_size=(224, 224),
        scale=8,
        context_length=77,
    )
    train_point = trainset[0]
    # Convert tensors to PIL images and save them
    # for i, (image_lr, image_hr) in enumerate(zip(train_point["image_lr"], train_point["image_hr"])):
    image_lr = train_point["image_lr"]
    image_hr = train_point["image_hr"]
    
    img = T.ToPILImage()(image_lr)
    img.save(f"tmp/train_lr.png")
    img = T.ToPILImage()(image_hr)
    img.save(f"tmp/train_hr.png")
    
    

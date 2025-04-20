from typing import Any, Dict, Tuple, List, Optional, Literal

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.image import PeakSignalNoiseRatio

from pytorch_lightning.loggers import WandbLogger

from diffusers import DDPMScheduler

from src.losses.clip_sim_loss import CLIPSimilarityLoss


# torch.autograd.set_detect_anomaly(True)


def denormalize(
    image: torch.Tensor,
    mean: List[float] = (0.48145466, 0.4578275, 0.40821073),
    std: List[float] = (0.26862954, 0.26130258, 0.27577711),
) -> torch.Tensor:
    dtype = image.dtype
    device = image.device
    mean = torch.as_tensor(mean, dtype=dtype, device=device)
    std = torch.as_tensor(std, dtype=dtype, device=device)

    return image * std[:, None, None] + mean[:, None, None]


class TSRBaseModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        diff_steps: int = 1000,
        txt2img_similarity: bool = True,
        img2img_similarity: bool = True,
        beta_txt2img: float = 0.1,
        beta_img2img: float = 0.1,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # Noise scheduler for the diffusion model
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=diff_steps, beta_schedule="squaredcos_cap_v2"
        )

        # loss function
        self.recon_criterion = torch.nn.MSELoss()

        if txt2img_similarity:
            self.txt2img_similarity_criterion = CLIPSimilarityLoss(
                sim_type="txt2img", clip_model=self.net.clip_model
            )
        if img2img_similarity:
            self.img2img_similarity_criterion = CLIPSimilarityLoss(
                sim_type="img2img", clip_model=self.net.clip_model
            )
        # metric objects for calculating and averaging accuracy across batches
        # self.train_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.test_psnr = PeakSignalNoiseRatio(data_range=1.0)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        # self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_psnr_best = MaxMetric()

    def get_clean_estimate(
        self,
        predicted_noise: torch.Tensor,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Get the clean estimate of the input image.
        :param predicted_noise: The predicted noise from the model.
        :param noisy_input: The noisy input image.
        :param timesteps: The timesteps used for the diffusion process.
        :return: The clean estimate of the input image.
        """
        B = noisy_input.shape[0]

        alpha_cumprod = self.noise_scheduler.alphas_cumprod[timesteps].to(self.device)
        sqrt_alpha_prod = torch.sqrt(alpha_cumprod).view(B, 1, 1, 1)
        sqrt_one_minus_alpha_prod = torch.sqrt(1 - alpha_cumprod).view(B, 1, 1, 1)

        # Get the clean estimate of the input
        return (
            noisy_input - sqrt_one_minus_alpha_prod * predicted_noise
        ) / sqrt_alpha_prod

    def inference_step(self, batch) -> None:
        image_lr, input_ids, attention_mask, image_hr = (
            batch[k] for k in ("image_lr", "input_ids", "attention_mask", "image_hr")
        )

        x = torch.randn_like(image_lr).to(self.device)
        for i, t in enumerate(self.noise_scheduler.timesteps):
            # Get the predicted noise
            residual = self.forward(
                x=x,
                t=t,
                lr_image=image_lr,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            x = self.noise_scheduler.step(residual, t, x).prev_sample

        # denormalize images for psnr calculation
        reconstructed = image_lr + x
        reconstructed = denormalize(reconstructed).clamp(0, 1)
        image_hr = denormalize(image_hr).clamp(0, 1)

        return reconstructed, image_hr

    def forward(self, **kwargs) -> torch.Tensor:
        """Perform a single forward pass through the network."""
        return self.net(**kwargs)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        # self.val_loss.reset()
        self.val_psnr.reset()
        self.val_psnr_best.reset()

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """

        image_lr, input_ids, attention_mask, image_hr = (
            batch[k] for k in ("image_lr", "input_ids", "attention_mask", "image_hr")
        )

        # device = self.device
        B = image_lr.shape[0]

        input = image_hr - image_lr

        noise = torch.randn_like(image_lr).to(self.device)
        timesteps = torch.randint(
            0, self.hparams.diff_steps, (B,), dtype=torch.int64, device=self.device
        )
        noisy_input = self.noise_scheduler.add_noise(input, noise, timesteps)

        # loss, preds, targets = self.model_step(batch)
        predicted_noise = self.forward(
            x=noisy_input,
            t=timesteps,
            lr_image=image_lr,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        recon_loss = self.recon_criterion(predicted_noise, noise)

        loss = recon_loss
        if self.hparams.img2img_similarity or self.hparams.txt2img_similarity:
            refined_image = image_lr + self.get_clean_estimate(
                predicted_noise, noisy_input, timesteps
            )
            if self.hparams.txt2img_similarity:
                loss += self.hparams.beta_txt2img * self.txt2img_similarity_criterion(
                    x1=input_ids,
                    x2=refined_image,
                    attention_mask=attention_mask,
                )
            if self.hparams.img2img_similarity:
                loss += self.hparams.beta_img2img * self.img2img_similarity_criterion(
                    x1=image_hr, x2=refined_image
                )

        # update and log metrics
        self.train_loss(loss)

        self.log(
            "train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True
        )

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """

        reconstructed, targets = self.inference_step(batch)
        self.val_psnr(reconstructed, targets)
        # loss, preds, targets = self.model_step(batch)

        # update and log metrics
        # self.val_loss(loss)
        # self.val_psnr(preds, targets)
        # self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.log("val/psnr", self.val_psnr, on_step=True, on_epoch=True, prog_bar=True)

        if (
            batch_idx
            == 0
            # and isinstance(self.logger, WandbLogger)
        ):
            # Only Log 16 images at max
            max_images_logs = 16
            if len(targets) < max_images_logs:
                max_images_logs = len(targets)

            # Log the reconstructed images
            self.logger.log_image(
                key="val/pred_image",
                images=list(reconstructed[:max_images_logs]),
            )

            # Log the ground truth images
            self.logger.log_image(
                key="val/gt_image",
                images=list(targets[:max_images_logs]),
            )

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        psnr = self.val_psnr.compute()  # get current val psnr
        self.val_psnr_best(psnr)  # update best so far val psnr
        # log `val_psnr_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            "val/psnr_best", self.val_psnr_best.compute(), sync_dist=True, prog_bar=True
        )

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """

        reconstructed, targets = self.inference_step(batch)
        self.test_psnr(reconstructed, targets)
        # loss, preds, targets = self.model_step(batch)

        # # update and log metrics
        # self.test_loss(loss)
        # self.test_psnr(preds, targets)
        # self.log(
        #     "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        # )
        self.log(
            "test/psnr", self.test_psnr, on_step=True, on_epoch=True, prog_bar=True
        )

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    # "monitor": "val/psnr",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    model = TSRBaseModule()
    # for testing purposes

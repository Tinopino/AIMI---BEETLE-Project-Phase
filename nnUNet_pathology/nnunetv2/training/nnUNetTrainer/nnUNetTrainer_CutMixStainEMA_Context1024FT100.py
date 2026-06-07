import torch

from .nnUNetTrainer_CutMixStainEMA import nnUNetTrainer_CutMixStainEMA


class nnUNetTrainer_CutMixStainEMA_Context1024FT100(
    nnUNetTrainer_CutMixStainEMA
):
    """
    Fine-tunes a completed CutMix + stain-jitter + EMA model using a larger
    1024x1024 WholeSlideData context window.

    Inherited:
    - alpha-weighted Dice + focal loss;
    - CutMix;
    - stain jitter;
    - EMA;
    - general-best and class-specific-best checkpoint saving;
    - checkpoint_latest.pth and checkpoint_final.pth;
    - per-class metric logging.

    Changed:
    - patch size: 1024x1024;
    - online WSD batch size: 2;
    - initial learning rate: 0.0005;
    - additional epochs: 100.
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plans,
            configuration,
            fold,
            dataset_json,
            unpack_dataset,
            device,
        )

        expected_patch_size = [1024, 1024]
        actual_patch_size = list(self.configuration_manager.patch_size)

        if actual_patch_size != expected_patch_size:
            raise RuntimeError(
                "Context trainer requires patch_size="
                f"{expected_patch_size}, received {actual_patch_size}. "
                "Use configuration='2d_context1024'."
            )

        self.wsd_batch_size_override = 2
        self.initial_lr = 5e-4
        self.num_epochs = 100
        self.save_every = 1

        self.print_to_log_file(
            "Using CutMix + stain jitter + EMA context fine-tuning: "
            "patch_size=[1024, 1024], WSD batch_size=2, "
            "initial_lr=0.0005, num_epochs=100",
            also_print_to_console=True,
        )

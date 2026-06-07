
# =============================================================================
# TARGETED HARD-EXAMPLE OVERSAMPLING: WEIGHTED FOCAL + HARD 2<->3 MINING
# =============================================================================

class nnUNetTrainerPathologyWFCHardMining250(
    nnUNetTrainerPathologyFocalClassMetricsAlpha
):
    """
    Fresh 250-epoch weighted-focal run with confusion-aware hard-example sampling.

    The weighted-focal loss, architecture, augmentations, and base WholeSlideData
    label sampling remain unchanged. During TRAINING only, 25% of patch centers
    are drawn from a manifest mined from class-2 <-> class-3 errors made by the
    prior 250-epoch weighted-focal model. Validation remains unchanged.

    Environment variables:
        HARD_MINING_MANIFEST
        HARD_MINING_FRACTION   default: 0.25
        HARD_MINING_JITTER     default: 128
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

        self.hard_mining_manifest = os.environ.get(
            "HARD_MINING_MANIFEST",
            "/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/"
            "pathology/hard_mining/wf250_fold0_train_hard_confusions.csv",
        )
        self.hard_mining_fraction = float(
            os.environ.get("HARD_MINING_FRACTION", "0.25")
        )
        self.hard_mining_jitter = int(
            os.environ.get("HARD_MINING_JITTER", "128")
        )

        if not os.path.isfile(self.hard_mining_manifest):
            raise FileNotFoundError(
                "Hard-mining manifest does not exist: "
                f"{self.hard_mining_manifest}"
            )

        self.print_to_log_file(
            "Using weighted focal + targeted hard-example mining: "
            f"manifest={self.hard_mining_manifest}, "
            f"fraction={self.hard_mining_fraction}, "
            f"jitter={self.hard_mining_jitter}",
            also_print_to_console=True,
        )

    def modify_fill_template(self, fill_template):
        """
        Replace the default WholeSlideData BatchReferenceSampler.

        The custom sampler itself checks dataset.mode and applies hard examples
        only in training mode. The copied validation config therefore remains a
        normal validation sampler.
        """
        super().modify_fill_template(fill_template)

        fill_template["batch_reference_sampler"] = {
            "*object": (
                "nnunetv2.training.nnUNetTrainer.variants.pathology."
                "hard_mining_batch_reference_sampler."
                "HardMiningBatchReferenceSampler"
            ),
            "dataset": "${dataset}",
            "batch_size": "${batch_shape.batch_size}",
            "label_sampler": "${label_sampler}",
            "annotation_sampler": "${annotation_sampler}",
            "point_sampler": "${point_sampler}",
            "manifest_path": self.hard_mining_manifest,
            "hard_fraction": self.hard_mining_fraction,
            "jitter": self.hard_mining_jitter,
            "seed": "${seed}",
        }

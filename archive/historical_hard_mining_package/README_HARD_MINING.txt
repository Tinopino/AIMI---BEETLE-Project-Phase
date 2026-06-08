BEETLE weighted-focal + targeted hard-example mining
====================================================

Goal
----
Mine class-2 <-> class-3 errors from TRAINING slides using the completed
250-epoch weighted-focal model, then train a fresh 250-epoch weighted-focal
model in which 25% of training patch centers come from those hard regions.

The validation sampler is unchanged. Do not mine from validation slides.

Install
-------
On your local machine, copy this package to the cluster repo and extract it.
Then run:

    cd ~/AIMI---BEETLE-Project-Phase
    bash <EXTRACTED_PACKAGE_DIR>/install_hard_mining_files.sh

Create fold-0 training CSV
--------------------------
    cd ~/AIMI---BEETLE-Project-Phase

    python make_fold_training_inference_csv.py \
      --splits-json /vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_preprocessed/Dataset301_BEETLE/splits.json \
      --fold 0 \
      --subset training \
      --out-csv /home/tijnveldwijk/fold0_training_inference_inputs.csv

    head -5 /home/tijnveldwijk/fold0_training_inference_inputs.csv

Important:
- The mining script expects TIF/TIFF masks readable through TiffSlide.
- If the helper prints a warning about non-TIFF annotation paths, stop before
  queuing mining and inspect the CSV generation approach.

Mine hard regions from the completed weighted-focal model
---------------------------------------------------------
Mirroring is explicitly enabled.

    cd ~/AIMI---BEETLE-Project-Phase
    sbatch run_mine_wf250_hard_manifest.slurm

After it finishes:

    MANIFEST=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/hard_mining/wf250_fold0_train_hard_confusions.csv
    wc -l "$MANIFEST"
    head -5 "$MANIFEST"
    cat "${MANIFEST%.csv}.summary.json"

Train a fresh 250-epoch hard-mining model
----------------------------------------
    cd ~/AIMI---BEETLE-Project-Phase
    sbatch run_pathology_wfhardmine250.slurm 0 fresh

Evaluate the resulting general-best checkpoint with mirroring
-------------------------------------------------------------
    sbatch \
      --job-name=eval_wfhardmine250_mirror \
      --export=ALL,MODEL_BASE_PATH=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_results/Dataset301_BEETLE/nnUNetTrainerPathologyWFCHardMining250__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d,CHECKPOINT_NAME=checkpoint_best.pth,CHECKPOINT_TAG=wfhardmine250_best_mirror_visual,SAVE_VISUALS=1,USE_MIRRORING=1 \
      run_original_beetle_fast_eval.slurm

Interpretation
--------------
Compare against the mirrored 250-epoch weighted-focal model:
- macro and micro Dice;
- class-2 and class-3 Dice;
- row-normalized GT3 -> pred2 error;
- row-normalized GT2 -> pred3 error.

This is a clean ablation:
- same architecture;
- same weighted-focal loss;
- same 250-epoch budget;
- same standard sampler for 75% of training coordinates;
- only 25% targeted hard-example coordinates are added.

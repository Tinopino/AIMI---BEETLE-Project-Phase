# Historical cluster scripts

The exact iterative cluster workflow used during development is preserved in
Git history at commit:

`fbcb9477914e3d0db0b424cbf8f8a87f4bec2f49`

The cleaned repository intentionally removes the large collection of
one-off queue, repair, setup, and experiment-specific SLURM wrappers from the
current tree.

Use the stable top-level interface instead:

```bash
python experiments.py --show
python train.py --experiment cutmix_stain_ema --fold 0
python validate.py wsi --experiment context1024_ft100 --fold 0 --save-visuals
python validate.py aggregate --stage both
python validate.py external --experiment context1024_ft100
```

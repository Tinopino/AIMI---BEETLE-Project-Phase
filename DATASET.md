# BEETLE dataset access

The public BEETLE dataset is available from Zenodo:

- [BEETLE dataset record](https://doi.org/10.5281/zenodo.16812932)

Download and extract `annotations.zip` and `images.zip`, then set
`BEETLE_DATA_ROOT` in `paths.env` to the directory that contains `annotations/`
and `images/`.

The release contains development WSIs, evaluation ROIs, JSON annotations, and
rasterized TIFF masks. The TIFF masks are used only for held-out WSI evaluation.

On the course cluster, this project used:

```text
/vol/csedu-nobackup/course/IMC037_aimi/group14/aalina
```

Users with read permission can reuse that mirror. Otherwise, download the
public release and adjust `BEETLE_DATA_ROOT`.

After configuring the dataset root, rewrite the committed WSI split paths:

```bash
python configure_data_paths.py \
    --data-root "${BEETLE_DATA_ROOT}" \
    --nnunet-preprocessed "${nnUNet_preprocessed}"
```

Generate held-out WSI validation manifests with:

```bash
python configure_validation_inputs.py
```

The Zenodo record notes that TIGER WSIs must be downloaded separately from the
linked AWS Open Data source when reproducing outside the course cluster.

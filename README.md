# CFC-MCRF

# PRISMA Neural Fusion Experiments

This repository contains script versions of two PRISMA neural experiments:

- **FCN fusion model** for PAN/PRISMA + hyperspectral inputs
- **Cross-modal attention / ViT-style fusion model** for the same dataloader and preprocessing pipeline

The private imagery, clusters, activations, checkpoints, and `.mat` files are intentionally **not** included in the repository. Follow the folder layout below to rebuild the project locally.

## Repository layout

```text
.
├── prisma_models/
│   ├── dataloader.py          # shared PRISMA dataset and split loading
│   ├── models_fcn.py          # FCN model
│   ├── models_vit.py          # cross-modal attention / ViT-style model
│   ├── train.py               # shared training and prediction loops
│   └── utils.py               # shared image, raster, metrics, sliding-window helpers
├── scripts/
│   ├── train_fcn.py
│   ├── train_vit.py
│   ├── predict.py
│   └── extract_activations.py
├── matlab/
│   ├── multires_CFCCRF.m
│   └── utils_debug.m
├── configs/
│   ├── fcn.example.yaml
│   ├── vit.example.yaml
│   ├── predict.example.yaml
│   └── extract_activations.example.yaml
├── data/
│   ├── raw/                   # private image tiles; not tracked by git
│   ├── processed/             # optional preprocessed rasters; not tracked
│   ├── external/              # optional external files; not tracked
│   ├── mat/
│   │   ├── clusters/          # private MATLAB cluster .mat files
│   │   └── activations/       # private activation/posterior .mat files
│   └── splits/                # train/test tile id lists
└── outputs/
    ├── checkpoints/           # trained PyTorch weights; not tracked
    ├── predictions/           # predicted labels/posteriors; not tracked
    ├── activations/           # exported .mat activations; not tracked
    └── logs/
```

## Data layout

Place each PRISMA tile in its own folder under `data/raw/`. The folder name must match the tile id used in `data/splits/train.txt` and `data/splits/test.txt`.

Example:

```text
data/raw/
└── ER_001/
    ├── ER_001_Cube.tif
    ├── ER_001_VNIR_SWIR.tif
    └── ER_001_gt_CRS_registered.tif
```

By default, the dataloader expects these suffixes:

| File | Meaning |
|---|---|
| `<tile_id>_Cube.tif` | high-resolution PAN / PRISMA input |
| `<tile_id>_VNIR_SWIR.tif` | hyperspectral VNIR+SWIR input |
| `<tile_id>_gt_CRS_registered.tif` | registered ground-truth labels |

If your files have different names, edit the `pan_suffix`, `hsi_suffix`, and `label_suffix` entries in the YAML config.

## MATLAB multiresolution fusion files

The MATLAB part is intentionally separated from the Python neural models. Run it after the neural scripts have produced the activation/posterior `.mat` files.

### Execution order

1. **`matlab/utils_debug.m`**

   This is the MATLAB setup script. It creates the variables and configuration structures required by the fusion routine:

   - loads fine-resolution PAN activations and posteriors
   - loads coarse-resolution HYS activations and posteriors
   - loads the fine and coarse ground truth maps
   - builds `posteriors`, `imageTensor`, `Patch_division`, and `Param`
   - computes the fine/coarse cluster structures `Cluster_Data.f` and `Cluster_Data.c`
   - saves the generated cluster data for later reuse

2. **`matlab/multires_CFCCRF.m`**

   This is the multiresolution CFC-CRF execution script. It expects the variables prepared by `utils_debug.m` to already exist in the MATLAB workspace. It then:

   - splits the fine and coarse images into corresponding patches
   - extracts activations, posteriors, and ground truth for each patch
   - builds the fine-resolution and coarse-resolution CRF graphs
   - adds cluster-level connections at both resolutions
   - combines the fine/coarse energies into a multiresolution graph
   - runs graph cut with `GCMex`
   - saves patch-level results and elapsed times when `Param.Save.save_data = true`

### Expected MATLAB data layout

Large MATLAB files should not be committed to GitHub. Users should create the folders below locally and place their private files there.

```text
PRISMA_Tensors/
└── <dataset>/
    ├── gt.mat
    ├── gt_c.mat
    └── <CNN_model>/
        ├── <dataset>_<CNN_model>_act_8_PAN.mat
        ├── <dataset>_<CNN_model>_act_8_HYS.mat
        ├── <dataset>_<CNN_model>_post6_PAN.mat
        └── <dataset>_<CNN_model>_post6_HYS.mat

PRISMA_Clusters/
└── <dataset>/
    └── <CNN_model>/
        └── Clusters_img_<dataset>_<CNN_model>_<nFine>clF_<nCoarse>clC.mat

res/
└── <dataset>/
    └── <CNN_model>/
        └── ... saved MATLAB fusion outputs ...
```

The default values inside `utils_debug.m` are:

```matlab
dataset = 'SP';
CNN_model = 'FCN_SS';
```

Change these variables if your folder names or model names are different. The script builds paths relative to the repository location, so check `main_folder`, `save_dir`, `cluster_dir`, `data_dir`, and `gt_dir` before running.

### Required variables created by `utils_debug.m`

`multires_CFCCRF.m` assumes the following variables are already available in the MATLAB workspace:

| Variable | Purpose |
|---|---|
| `posteriors.f` | fine-resolution PAN softmax posteriors |
| `posteriors.c` | coarse-resolution HYS softmax posteriors |
| `imageTensor.feature_f` | fine-resolution PAN activations/features |
| `imageTensor.feature_c` | coarse-resolution HYS activations/features |
| `imageTensor.gt_f` | fine-resolution ground truth |
| `imageTensor.gt_c` | coarse-resolution ground truth |
| `Patch_division` | patch size, border, and patch indexing configuration |
| `Param` | save options, tensor options, clustering options, and loading flags |
| `Cluster_Data.f` | fine-resolution cluster data |
| `Cluster_Data.c` | coarse-resolution cluster data |

### Running the MATLAB routine

From MATLAB, run:

```matlab
cd matlab
utils_debug
multires_CFCCRF
```

If `Param.Save.save_data` is true, results are written under a numbered folder inside `res/`.

## Splits

Edit:

```text
data/splits/train.txt
data/splits/test.txt
```

Each line should contain one tile id, without file extensions:

```text
ER_001
ER_002
ER_003
```

These ids must match folders under `data/raw/`.

## Installation

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

For GeoTIFF/ENVI data, `GDAL` may be required. Install it using your system package manager or conda if pip installation fails.

## Train the FCN model

Edit `configs/fcn.example.yaml`, then run:

```bash
python scripts/train_fcn.py --config configs/fcn.example.yaml
```

The final checkpoint is saved in:

```text
outputs/checkpoints/fcn_prisma_final.pt
```

## Train the cross-modal attention / ViT-style model

Edit `configs/vit.example.yaml`, then run:

```bash
python scripts/train_vit.py --config configs/vit.example.yaml
```

The final checkpoint is saved in:

```text
outputs/checkpoints/vit_prisma_final.pt
```

## Predict labels for one tile

Edit `configs/predict.example.yaml`, then run:

```bash
python scripts/predict.py --config configs/predict.example.yaml
```

Predictions are written to `outputs/predictions/` unless changed in the config.

## Export activations/posteriors for MATLAB

Edit `configs/extract_activations.example.yaml`, then run:

```bash
python scripts/extract_activations.py --config configs/extract_activations.example.yaml
```

The script saves `.mat` files such as:

```text
outputs/activations/<tile_id>/posteriors_pan.mat
outputs/activations/<tile_id>/posteriors_hsi.mat
```

Copy or symlink them to `data/mat/activations/` if the MATLAB routines should consume them directly.

## GitHub note

The `.gitignore` excludes private and heavy files:

- raw imagery
- processed rasters
- `.mat` files
- PyTorch checkpoints
- predictions and logs

Only the code, configs, folder placeholders, and README should be uploaded to GitHub.

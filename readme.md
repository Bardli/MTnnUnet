

# Multi‑Task nnUNetv2 for Pancreas **Segmentation** and **Subtype Classification** in 3D CT

> Official code release for **“Deep Learning for Automatic Cancer Segmentation and Classification in 3D CT Scans.”**
> A shared 3D encoder (ResNet‑M) with two heads: **segmentation** and **subtype classification**.
> The system is implemented on top of **nnUNetv2** with ROI‑aware test‑time augmentation (TTA) and produces both `.nii.gz` segmentations and a `subtype_results.csv` for leaderboard submission. The schematic with a **shared image encoder + two heads** appears in the *figure on page 1 of the assignment document*. 


---

## Environment & Requirements

**Tested OS:** Ubuntu 20.04/22.04 and Windows 10/11
**Recommended hardware:** ≥8 CPU cores, 32 GB RAM, NVIDIA GPU (≥12 GB VRAM)
**CUDA:** 11.8 or 12.x (match your PyTorch)
**Python:** 3.9–3.11
**PyTorch:** ≥2.1 (uses `torch.compile`/`torch._dynamo`)

Install with `pip`:

```bash
# (optional) create a fresh environment
conda create -n pancreas-mtl python=3.10 -y
conda activate pancreas-mtl

# install CUDA-specific torch first (adjust version for your system)
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu121

# repo deps
pip install -r requirements.txt

# editable install (so console scripts are available)
pip install -e .
```

**What’s in `requirements.txt` (minimal):**

```
nnunetv2>=2.4
tqdm
numpy
acvl-utils
batchgenerators
torch>=2.1
```

---

## Dataset

This project uses cropped **pancreas CT ROIs** so it runs on free GPUs (Colab T4 / Kaggle P100). *The handout specifies that the original large 3D scans were cropped to smaller ROIs.* 

**Splits (training/validation counts by subtype):**
Train: **62 / 106 / 84**, Validation: **9 / 15 / 12** for **Subtype 0 / 1 / 2**, respectively. 

**Folder structure (as provided):** 

```
data
├── train
│   ├── subtype0
│   │   ├── quiz_0_041.nii.gz           # mask (0=background; 1=pancreas; 2=lesion)
│   │   ├── quiz_0_041_0000.nii.gz      # image
│   ├── subtype1
│   └── subtype2
├── validation
│   ├── subtype0
│   │   ├── quiz_0_168.nii.gz
│   │   ├── quiz_0_168_0000.nii.gz
│   ├── subtype1
│   └── subtype2
└── test                                # only images
    ├── quiz_037_0000.nii.gz
    ├── quiz_045_0000.nii.gz
    └── ...
```




## Preprocessing

We rely on **nnUNetv2**’s out‑of‑the‑box pipeline (spacing, padding/cropping, normalization). The supplied dataset already provides **ROI crops**, which greatly reduces memory and runtime. 

Typical steps:

* **Cropping**: already applied (ROI data). 
* **Intensity normalization**: nnUNetv2 default per‑case normalization via plans.
* **Resampling**: as defined in `plans.json` (handled automatically).




## Training

This repo extends **nnUNetv2** with a **classification head** while keeping the standard training loop. The model exposes both **segmentation logits** and **classification logits**.



### 1) Convert to nnUNetv2 structure & plan

```bash
# set your nnUNet paths (example)
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results

# Suppose your dataset id is 999 and name is Dataset999_PancreasQuiz
# Prepare JSONs & copy NIfTI into the nnUNetv2 directory layout before this step.

nnUNetv2_plan_and_preprocess -d 002 -c 3d_fullres -pl nnUNetPlannerResEncM
```

### 2) Train (fold 0 as example)

```bash
# base trainer name stays "nnUNetTrainer" unless you created a custom subclass
nnUNetv2_train 002 3d_fullres nnUNetTrainer 5 -p nnUNetResEncUNetMPlans
```

**Tracking & reporting:** the assignment asks you to **log training/validation curves and metrics for both segmentation and classification via Weights & Biases (wandb)**—please enable `wandb` in your trainer or scripts. 

**On class imbalance / overfitting:** you must describe your strategies in the report (e.g., loss weighting, focal loss, oversampling, augmentation, early stopping/regularization); these are part of the mandatory checklist. 



---

## Inference

We provide two CLI entry points compatible with nnUNetv2:

### A) Predict **by dataset id** (preferred)


```bash
nnUNetv2_predict -i INPUT_FILE -o OUT_FILE -d DATA_ID -c 3d_fullres -p nnUNetResEncUNetMPlans -f 5 -chk  checkpoint_best_macro_f1.pth
```
**Outputs produced:**

* Segmentation NIfTI for each case (same basename as input).
* `classification_results.json` — raw classification logits/probabilities per case.
* `subtype_results.csv` — **exact submission format** with two columns: `Names, Subtype`.
  Example required by the handout:

  ```
  Names,Subtype
  quiz_037.nii.gz,0
  quiz_045.nii.gz,1
  quiz_047.nii.gz,2
  ```



**What the code does differently at inference (high‑level):**

* Sliding‑window prediction for **segmentation** (Gaussian‑weighted stitching).
* **Classification** is computed per patch, aggregated with **foreground‑probability weights**, and supports **ROI‑aware TTA**; final case‑level logits are exported (and `subtype_results.csv` is written automatically when `-o` is set).


---

## Evaluation

Compute **Dice** for:

* **Whole pancreas**: `label > 0` (labels 1 + 2).
* **Lesion**: `label == 2`.

Compute **macro‑F1** for **subtype classification**.


---

## Results



| Model                   | Whole‑Pancreas Dice ↑ | Lesion Dice ↑ | Macro‑F1 (Subtype) ↑ |
| ----------------------- | --------------------: | ------------: | -------------------: | 
| Our multi‑task nnUNetv2 |       0.9126 ± 0.0560 |0.6293 ± 0.2883| 0.8580641805793935   |



---


## Colab / Jupyter

Add a notebook that:

1. Installs this repo,
2. Downloads/places the dataset,
3. Runs `nnUNetv2_plan_and_preprocess`, `nnUNetv2_train`, and `nnUNetv2_predict`,
4. Produces `subtype_results.csv`.

---


## Contributing & License

Contributions are welcome (issues, pull requests). Please add unit tests where possible.
License: choose one that fits (e.g., **MIT** or **Apache‑2.0**). Add a `LICENSE` file to the repo.

---

## Acknowledgements

We thank the maintainers of **nnUNetv2** and the providers of public datasets and benchmarking guidance. See “Related Work” and references in the assignment (nnU‑Net, Metrics Reloaded, etc.). 






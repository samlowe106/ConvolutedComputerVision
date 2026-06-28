# Why the chest X-ray models underperformed (and how to fix it)

_Investigation notes, 2026-06-25._

## TL;DR

The full-resolution chest X-ray models aren't fundamentally worse learners than the `pneumonia_mnist` models — the **evaluation and model-selection were broken**, which made every model look bad and made tuning impossible. Two problems dominate:

1. The shipped **validation set was 16 images** (8 NORMAL + 8 PNEUMONIA), so `val_loss` was pure noise and the "best model" checkpoint was effectively random.
2. One copy of the class-weight helper was **inverted**, up-weighting the majority class and pushing models to predict "pneumonia" for everything.

`pneumonia_mnist` "just works" mostly because MedMNIST ships a real ~520-image validation split and pre-curated images — not because the task is easier.

## Symptom

The convnets reached 0.90+ **train** accuracy but looked terrible on validation.
In every chest-xray notebook the val metrics are pinned at noise:

```
val_accuracy: 0.5000 ... val_recall: 1.0000 - val_tn: 0.0000 - val_tp: 8.0000
```

`val_tn=0, val_tp=8` means the model predicted PNEUMONIA for all 16 val images,
so accuracy sat at exactly 0.5. When it moved, it jumped between 0.375 / 0.5625 /
0.8125 — because **each image is 6.25% of the score**.

## Root causes (ranked)

### 1. The 16-image validation set

`ModelCheckpoint(monitor='val_loss', save_best_only=True)` selected whichever
epoch got the luckiest draw on 16 images. There was no usable early-stopping
signal and no way to compare architectures. The confidence interval on a val
accuracy estimate (`±1.96·√(p(1-p)/n)` at p≈0.9) shows why no amount of
"slightly bigger" helps:

| val size            | 95% CI on val accuracy |
| ------------------- | ---------------------- |
| 16 (original)       | ±15%                   |
| 48 (3×)             | ±8.5%                  |
| ~790 (15% of train) | ±2.1%                  |

Model selection compares candidates that differ by 1–3%. The noise band has to
sit *below* that, which means hundreds of images, not dozens.

### 2. Inverted class weights

The two helper variants disagree:

- `total / (2*count)` (inverse frequency) → `normal: 1.000, pneumonia: 0.346` ✅
  ([01_start.ipynb](./01_start.ipynb))
- `sum / len` (frequency) → `normal: 0.347, pneumonia: 1.000` ❌
  ([02_resizing.ipynb](./02_resizing.ipynb), [../pneumonia_mnist/Adam.ipynb](../pneumonia_mnist/Adam.ipynb))

The second form weights each class *proportional* to its frequency, so it
up-weights the majority (pneumonia) and reinforces the all-pneumonia collapse
seen in [03_transferring_and_resizing.ipynb](./03_transferring_and_resizing.ipynb)
(`recall: 1.0, tn: 0` for all 25 epochs). **Use the inverse-frequency form
everywhere.**

### 3. Patient leakage risk

Images are multi-per-patient: 3,875 pneumonia images span 1,635 `personN` ids
(~2.4 each), and NORMAL `IM-####` studies have up to 6 images. A naive
image-level split puts the same patient in train *and* val → optimistic val that
doesn't transfer. Splits must be **grouped by patient**.

### 4. Misleading metric

The test set is 62.5% pneumonia, so an all-pneumonia predictor scores 62.5%
"accuracy." Report **ROC-AUC / PR-AUC + confusion matrix**, not accuracy.

### 5. Shuffled evaluation corrupted the confusion matrices

`image_dataset_from_directory` defaults to **`shuffle=True`**, and the underlying
`tf.data` shuffle **reshuffles on every iteration**. The chest notebooks then read
the test set *twice, in separate passes*:

```python
y_true = np.concatenate([y for x, y in test_ds])  # pass 1 — one order
y_pred = np.round(model.predict(test_ds))          # pass 2 — a DIFFERENT order
confusion_matrix(y_true, y_pred)                    # labels vs predictions of different images
```

So every confusion matrix paired each image's label with *some other image's*
prediction — it measured the model against noise, which inflates the off-diagonal
(false positives and false negatives) toward what random guessing produces. The
"lots of false negatives" first noticed in `04_gap` was largely this artifact.

What makes it sneaky: the per-epoch `val_recall` / `val_precision` printed by `fit`
are **correct** — Keras computes them per batch, so labels and predictions never
separate. Only the *separately reconstructed* confusion matrix is corrupted, so the
headline metrics look plausible while the CM lies. (A quick check: iterate the test
set twice and compare label orders — with the default shuffle they differ.)

**Fix:** evaluation sets must be deterministic. We set `shuffle=False` on the
`validation` and `test` loaders in all chest notebooks (train stays shuffled). For
aggregate metrics order is irrelevant; the point is that two passes now agree, so
`y_true` and `predict` line up. (Equivalently, gather labels and predictions in a
single pass.)

**Scope:** only the chest notebooks were affected — they build datasets with
`image_dataset_from_directory`. The `pneumonia_mnist` and `mnist` notebooks load
via `tfds` and add no `.shuffle()` to the test pipeline (tfds test splits iterate
deterministically), and `cifar-10` evaluates on in-memory NumPy arrays — all
already aligned.

### Smaller issues

- `Flatten → Dense(1024)` is ~40M params in one layer (~850 s/epoch at 150×150);
  a `GlobalAveragePooling2D` head is smaller, regularizes better, trains faster.
- `RandomFlip("horizontal")` is anatomically wrong for chest X-rays (heart is
  left-of-midline) — drop it.
- Filename mismatch: writes `best_model.keras` but loads
  `best_model_resizing_dense.keras` (a stale file).
- A single `ModelCheckpoint` is reused across `model_1..model_5` in one notebook.
- No `EarlyStopping` / `ReduceLROnPlateau` (these need a real val set first).

## Why `pneumonia_mnist` fits more easily

Same images conceptually, very different conditions:

1. **Real val split** (~520 images) → `val_loss` is a real signal, checkpointing
   works. This is the dominant factor.
2. **Curated inputs**: MedMNIST images are center-cropped, registered,
   intensity-normalized, and downsampled to 28×28 — exactly the nuisance variance
   (size, contrast, rotation, borders, burned-in annotations) the full-res net
   otherwise has to fight.
3. **Small inputs** fit the tiny dense nets without much overfitting room.
4. The class-weight bug hurts MNIST too, but doesn't tank it because pneumonia is
   also the test-set majority there.

## What was changed

- **Lint**: renamed the ambiguous `l` → `label` in `get_class_training_weights`
  and dropped an unused `ax` so `ruff check .` passes across all notebooks.
- **Regroup**: [split_data.py](./split_data.py) re-pools `train` + `val` and
  carves a **stratified, patient-grouped** validation set (~15%), leaving the
  official `test` set untouched for comparability.
- **Callbacks**: replaced the single reused `checkpoint_callback` with a
  `make_callbacks(filepath)` helper (ModelCheckpoint + `EarlyStopping` +
  `ReduceLROnPlateau`) in every training notebook. Each model now checkpoints to
  its own file and each `load_model` reads the matching file — fixing the
  `02_resizing` filename mismatch and the cross-model checkpoint collisions.
  Still monitors `val_loss`; switching to `val_auc` is the remaining piece.
- **Class weights**: replaced the inverted `sum/len` `get_class_training_weights`
  (which up-weighted the majority class) with proper inverse-frequency weighting
  in `02_resizing`, `03_transferring_and_resizing`, and both `pneumonia_mnist`
  notebooks. Now `normal: 1.000, pneumonia: 0.348` (minority up-weighted).
- **GAP head**: swapped `Flatten` → `GlobalAveragePooling2D` in the CNN models
  (12 cells), collapsing the ~42M-param dense head to ~0.5M. Left untouched: the
  pure-MLP baselines (no conv feature map) and `03_transferring_and_resizing`
  (it transfers dense-layer weights, so the head shape must stay fixed).
- **Deterministic evaluation**: set `shuffle=False` on the `validation`/`test`
  loaders in all five chest notebooks (`01_start`, `02_resizing`, `02_transferring`,
  `03_transferring_and_resizing`, `04_gap`). See _Root cause 5_ — the old confusion
  matrices compared labels to unrelated predictions because the loader reshuffled
  between the `y_true` pass and the `predict` pass.

Resulting split (deterministic, `seed=42`, `val-frac=0.15`):

| split | NORMAL | PNEUMONIA | total | %pneu |
| ----- | -----: | --------: | ----: | ----: |
| train |  1,146 |     3,296 | 4,442 | 74.2% |
| val   |    203 |       587 |   790 | 74.3% |
| test  |    234 |       390 |   624 | 62.5% |

Reproduce (idempotent — always re-pools train+val first):

```bash
uv run --no-project --with scikit-learn \
  python chest-x-ray-images-pneumonia/split_data.py            # apply
uv run --no-project --with scikit-learn \
  python chest-x-ray-images-pneumonia/split_data.py --dry-run  # preview only
```

The other datasets (`pneumonia_mnist`, `mnist`, `cifar-10`) load via tfds/keras
with proper splits and need no regrouping.

## Architectures explored

The notebooks walk a progression of approaches to the same binary task, each
building on the last:

| Notebook | Input | Approach | Idea |
| --- | --- | --- | --- |
| `01_start` | 150×150 | CNN from scratch (Xception-like), `Flatten → Dense` head | full-resolution baseline |
| `02_resizing` | 28×28 | CNN from scratch on downsampled images | shrink to the MedMNIST size |
| `02_transferring` | 150×150 | transfer ImageNet conv weights (VGG16/19) | borrow natural-image features |
| `03_transferring_and_resizing` | 28×28 | transfer conv weights from our pneumonia-MNIST net (13 iterations) | combine resize + transfer |
| `04_gap` | 28×28 | best model of `03` with a `GlobalAveragePooling2D` head | kill the dense-head param explosion |
| `05_attention` | 150×150 | hybrid CNN backbone → transformer encoder over the feature-map tokens | add self-attention without a data-hungry pure ViT |
| `06_pretrained` | 224×224 | DenseNet-121 (ImageNet), freeze → fine-tune | a strong pretrained backbone (CheXNet-style) |

Two takeaways from the arc:

- **Resolution is the quiet ceiling.** The 28×28 lineage (`02_resizing`, `03`,
  `04_gap`) exists only to reuse the pneumonia-MNIST transfer, but pneumonia signs
  barely survive that downsampling. The higher-resolution notebooks (`05`, `06`)
  give the model something to actually see.
- **On ~4.4k images, borrowed features beat learned-from-scratch ones.** A
  from-scratch net — CNN or transformer — is fighting the data budget; transferring
  from a large source (ImageNet, or ideally a chest-X-ray-pretrained net) is the
  highest-leverage move, which is why `06` is the most promising route.

Evaluation is shared and consistent across all of them: deterministic val/test
loaders, `val_auc`-based selection + early stopping, an F1-tuned decision
threshold, and the `visualization` helpers (`summary_graphics`,
`plot_binary_confusion_matrix`).

## Remaining next steps

1. Change the callback `monitor` from `val_loss` to **`val_auc` / PR-AUC** (the
   `EarlyStopping` + `ReduceLROnPlateau` + per-model checkpoints are already in).
2. Drop horizontal flip; report AUC + confusion matrix instead of accuracy.

_Done: real validation split, per-model checkpoints + early stopping,
inverse-frequency class weights, and the `GlobalAveragePooling2D` head swap._

## Why `GlobalAveragePooling2D` beats `Flatten → Dense`

The current head `Flatten → Dense(1024)` on a 150×150 feature stack is ~40M
parameters in a single layer — the overwhelming majority of the model and its
main overfitting surface. `GlobalAveragePooling2D` collapses each feature map to
its spatial mean, producing one number per channel (e.g. 512 values) with **zero
parameters**, so the classifier head shrinks from tens of millions of weights to
a few hundred.

Benefits:

- **Regularization** - no giant dense layer to memorize the training set; GAP
  acts as a structural regularizer (the original motivation in *Network in
  Network*).
- **Spatial robustness** - averaging over positions makes the head invariant to
  where a feature appears, instead of tying each pixel location to its own weight.
- **Feature-map ↔ concept correspondence** - GAP encourages each channel to act
  as a confidence map for a concept, which is also what makes class-activation
  maps (CAM) work for localization.

Primary reference: Lin, Chen & Yan, **"Network In Network"** (2013),
[arXiv:1312.4400](https://arxiv.org/abs/1312.4400) - section 3.2 introduces global
average pooling as a parameter-free replacement for fully-connected layers.
Follow-up worth reading: Zhou et al., **"Learning Deep Features for Discriminative
Localization"** (CVPR 2016), [arXiv:1512.04150](https://arxiv.org/abs/1512.04150),
which builds CAM on top of a GAP layer.

With a real validation set in place, these models should reach ~0.92–0.96 ROC-AUC
— the networks were already learning (train acc 0.90+); we just couldn't measure
or select them.

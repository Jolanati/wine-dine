# Beer Peer � Implementation Plan

## Step-by-step reference

> **Architecture in one sentence:** CNN classifies a food photo ? food label ? lookup table ? 2-3 beer styles ? BiLSTM retrieves flavor language per style; joint model learns food-beer compatibility from (image embedding, review embedding) pairs supervised by the lookup table.

---

## Product recap

A user photographs their food. Beer Peer returns three beer style recommendations, each with flavor language written by real beer drinkers. Three independent components: a CNN that sees the food, a BiLSTM that reads beer reviews, and a lookup table that connects them. The joint model (+10 pts) learns to generalize that connection � scoring food-beer pairs the lookup table never explicitly covered.

---

## Datasets

| Dataset | How to load | Content | Used for |
| --- | --- | --- | --- |
| Food-101 | `load_dataset("ethz/food101")` | 101,000 images, 101 food classes | CNN training |
| BeerAdvocate | `kaggle datasets download rdoume/beerreviews` | 1.5M reviews, `beer_style`, `review_text`, `review_overall` | BiLSTM training |
| Pairing table | Embedded CSV string in notebook | 101 foods � 3 beer styles | Lookup + joint model labels |

**BiLSTM labels:** 8 macro-style classes (Lager / IPA / Stout-Porter / Wheat / Sour-Farmhouse / Amber-Brown / Pale Ale / Specialty) � group the 100+ raw `beer_style` values before training.

---

## Development phases

| Phase | Environment | Sections | What happens |
| --- | --- | --- | --- |
| **Phase 1** | VS Code (CPU) | 1�4, 6, 10�12 | Data loading. EDA. Text preprocessing. LSTM + BiLSTM training. |
| **Phase 2** | Google Colab (GPU) | 5, 7�9, 13 | Image data loaders. CNN training (scratch + ResNet-50). Grad-CAM. Joint model training. |
| **Phase 3** | VS Code (CPU) | 9, 12, 14�15, deployment | Explainability. Recommendation card. Streamlit app. PDF export. |

---

## Before you start � one-time setup

1. Folders `weights/`, `figures/`, `deployment/` are created automatically by the notebook.
2. Open `beer-peer/beer_peer.ipynb`.
3. Run **Section 1** (pip installs) once per environment.
4. For BeerAdvocate: set up Kaggle API key (`~/.kaggle/kaggle.json`) before Phase 1.

---

## PHASE 1 � Local skeleton (VS Code, CPU)

> Goal: every cell runs without errors. Accuracy does not matter yet.

---

### STEP 1 � Load the datasets

**Section 2 of the notebook.**

```python
from datasets import load_dataset
ds_food = load_dataset("ethz/food101")   # downloads ~5 GB

import pandas as pd
df_beer = pd.read_csv("beerreviews.csv") # after kaggle download
```

- Print shapes, column names, class counts for both.
- Map `beer_style` ? 8 macro-style classes; print distribution.
- Load the pairing table from the embedded CSV string; display as a DataFrame.

**Done when:** three DataFrames visible with correct shapes.

---

### STEP 2 � EDA: image dataset

**Section 3 of the notebook.**

- 3�6 sample image grid (2 images per food class, randomly sampled from 18 classes).
- Class distribution bar chart (101 bars � confirm ~750 train / ~250 test per class).
- Image dimension histogram (confirm variable resolution ? need resize in preprocessing).

**Done when:** sample grid and distribution chart visible.

---

### STEP 3 � EDA: text dataset

**Section 4 of the notebook.**

- Review length histogram (words per review); mark the 95th percentile ? set as max token length.
- Macro-style class distribution bar chart (confirm reasonable balance after grouping).
- Word cloud per macro-style (8 clouds).
- 3 sample reviews per macro-style.

**Done when:** all plots visible; max token length chosen.

---

### STEP 4 � Text preprocessing and data loaders

**Section 6 of the notebook.**

- Tokenise `review_text` at word level; lowercase; strip punctuation.
- Load GloVe-100d; build vocab from training split only (no data leakage).
- Pad/truncate to 95th-percentile token length; print % truncated.
- Train / val / test split: 70 / 15 / 15, stratified by macro-style, `SEED=42`.
- Create `DataLoader` (batch size 64).

**Done when:** one batch prints `torch.Size([64, N])`.

---

### STEP 5 � CNN skeleton (architecture only)

**Section 7 of the notebook.**

- Define 3-block custom CNN: `Conv2d ? BatchNorm ? ReLU ? MaxPool2d` � 3 ? `Flatten ? FC(512) ? FC(101)`.
- `model(torch.randn(2, 3, 224, 224))` returns `[2, 101]`.
- Full training in Phase 2 only.

**Done when:** forward pass runs; parameter count printed.

---

### STEP 6 � ResNet-50 skeleton

**Section 8 of the notebook.**

- `torchvision.models.resnet50(weights='IMAGENET1K_V1')`; freeze all; replace `fc` with `nn.Linear(2048, 101)`.
- Forward pass returns `[2, 101]`.

**Done when:** shape correct; frozen vs. trainable parameter counts printed.

---

### STEP 7 � LSTM baseline (full training on CPU)

**Section 10 of the notebook.**

- `nn.Embedding(GloVe) ? nn.LSTM(hidden=128) ? nn.Linear(128, 8)`.
- Train 3 epochs; plot loss + accuracy curves.
- Report val accuracy.

**Done when:** loss decreases across epochs.

---

### STEP 8 � BiLSTM with attention (full training on CPU)

**Section 11 of the notebook.**

- `bidirectional=True`, hidden 128 (output 256).
- Attention: `score = softmax(W � h_t)` ? weighted sum ? classification head.
- Train 3 epochs; compare val accuracy vs. LSTM baseline.
- Save `weights/bilstm.pt`.

**Done when:** BiLSTM trained, weights saved, accuracy comparison written in markdown.

---

### STEP 9 � Joint model skeleton

**Section 13 of the notebook.**

Build the training data from what you already have:

```python
# Positive pairs: food image + review of a paired style ? label 1
# Negative pairs: food image + review of a non-paired style ? label 0
# Source of truth: food_beer_pairing.csv

def build_pairs(df_food, df_beer, pairing_df, n_negatives_per_positive=3):
    pairs = []
    for _, row in pairing_df.iterrows():
        food = row['food']
        paired_styles = [row['beer_style_1'], row['beer_style_2'], row['beer_style_3']]
        food_images = df_food.filter(label=food101_label_map[food])
        
        for style in paired_styles:
            reviews = df_beer[df_beer['macro_style'] == style]['review_text'].sample(5)
            for img in food_images.select(range(5)):
                for rev in reviews:
                    pairs.append((img, rev, 1))  # positive
        
        all_styles = set(MACRO_STYLES) - set(paired_styles)
        for style in random.sample(all_styles, n_negatives_per_positive):
            reviews = df_beer[df_beer['macro_style'] == style]['review_text'].sample(5)
            for img in food_images.select(range(5)):
                for rev in reviews:
                    pairs.append((img, rev, 0))  # negative
    return pairs
```

Architecture (skeleton � forward pass on dummy tensors only):

```python
class CompatibilityModel(nn.Module):
    def __init__(self, cnn_encoder, bilstm_encoder):
        super().__init__()
        self.cnn = cnn_encoder    # frozen, outputs 512-d (scratch) or 2048-d (ResNet)
        self.bilstm = bilstm_encoder  # frozen, outputs 256-d
        self.fc = nn.Sequential(
            nn.Linear(512 + 256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, image, text):
        img_emb = self.cnn(image)    # [B, 512]
        txt_emb = self.bilstm(text)  # [B, 256]
        x = torch.cat([img_emb, txt_emb], dim=1)  # [B, 768]
        return self.fc(x)            # [B, 1]
```

- `model(torch.randn(2,3,224,224), torch.randint(0,1000,(2,N)))` returns `[2, 1]`.
- Full training in Phase 2.

**Done when:** forward pass runs; only FC head parameters are trainable.

---

## PHASE 2 � Real training (Google Colab, GPU)

> Goal: get good weights. Only this phase requires GPU.

---

### STEP 10 � Move to Colab

- Upload `beer_peer.ipynb` to Colab; mount Drive; set weight output paths to Drive.
- Run **Section 5**: build image `DataLoader` from Food-101.
  - Transforms: `Resize(256) ? CenterCrop(224) ? RandomHorizontalFlip ? ColorJitter ? ToTensor ? Normalize(ImageNet stats)`.
  - Train / val / test split stratified by class (750/250 already done by Food-101 � use official split).
  - Confirm one batch: `torch.Size([B, 3, 224, 224])`.

---

### STEP 11 � Train CNN scratch (full)

**Section 7 � full training.**

- 20 epochs, early stopping on val loss (patience 3).
- Optimizer: Adam lr 1e-3; scheduler: ReduceLROnPlateau.
- Save best to `weights/cnn_scratch.pt`.
- Report test: Accuracy (top-1 and top-5), macro F1, Confusion Matrix (sample 20 classes for readability).

---

### STEP 12 � Train ResNet-50 (full)

**Section 8 � full training.**

- Phase A: frozen backbone, train FC head only, 5 epochs, lr 1e-3.
- Phase B: unfreeze last ResNet block (`layer4`), lr 1e-4, 10 more epochs.
- Save best to `weights/cnn_resnet50.pt`.
- Report test: Accuracy, macro F1, Confusion Matrix.
- Markdown: compare scratch vs. ResNet-50 � accuracy, training time, top-5 failure modes.

---

### STEP 13 � Train joint model (full)

**Section 13 � full training.**

- Load frozen `cnn_resnet50.pt` encoder (replace FC with identity to get 2048-d embedding).
- Load frozen `bilstm.pt` encoder (remove classification head to get 256-d embedding).
- Build compatibility pairs (Step 9 code, now on full data).
- Train only the FC head, 10 epochs, BCELoss, Adam lr 1e-3.
- Save to `weights/joint_model.pt`.
- Report: AUC-ROC, precision, recall on test pairs.
- **The interesting experiment:** run the joint model on food-style pairs NOT in the lookup table. Print the top-5 unexpected high-scoring pairs � these are the "learned extensions" of the hardcoded rules.

---

### STEP 14 � Download artefacts

Download from Colab / Drive to local `weights/`:

- `cnn_scratch.pt`
- `cnn_resnet50.pt`
- `bilstm.pt`
- `joint_model.pt`

---

## PHASE 3 � Analysis, explainability, deployment (VS Code, CPU)

> Goal: rubric-required analysis and working demo.

---

### STEP 15 � CNN explainability (Grad-CAM)

**Section 9 of the notebook.**

- Load `cnn_resnet50.pt`; select 6 test images (2 correct, 2 wrong class, 2 top-5 correct but top-1 wrong).
- Generate Grad-CAM heatmap with `torchcam`; display original + overlay side by side.
- 3-sentence markdown per example: which part of the food drove the classification?

---

### STEP 16 � BiLSTM explainability (attention weights)

**Section 12 of the notebook.**

- Load `bilstm.pt`; pick 6 reviews (1 per macro-style, 6 of 8).
- Extract attention weight vector; render as colour-highlighted text (word opacity ? weight).
- 3-sentence interpretation: which words ("hoppy", "roasty", "crisp") drove each style prediction?

---

### STEP 17 � Recommendation card + 20-example table

**Section 14 of the notebook.**

```python
def recommend(image_path):
    # 1. CNN ? food label (top-3 with confidence)
    food_label, confidence = cnn_predict(image_path)
    
    # 2. Lookup table ? 2-3 beer styles
    styles = get_beer_styles(food_label, pairing_df)
    
    # 3. Joint model ? re-score and re-rank styles
    joint_scores = {s: joint_score(image_path, sample_review(s)) for s in styles}
    ranked_styles = sorted(joint_scores, key=joint_scores.get, reverse=True)
    
    # 4. BiLSTM ? retrieve top-rated review sentences per style
    flavor_text = {s: get_flavor_language(s, df_beer) for s in ranked_styles}
    
    return format_card(food_label, confidence, ranked_styles, joint_scores, flavor_text)
```

- Run on 20 Food-101 test images.
- Display as table: Food | CNN confidence | Style 1 | Style 2 | Style 3 | Joint scores.
- Highlight 3 cases where joint model re-ranked the lookup table order � explain why.

---

### STEP 18 � Business framing (markdown cells)

**Section 15 of the notebook.**

Three markdown cells:

1. **Business framing** � target user (restaurant guests, home cooks, event planners). Decision supported: what beer to order or stock. Cost of a bad recommendation: wasted purchase, disappointed guest. Cost of a missed pairing: guest defaults to habit instead of trying something new.
2. **Ethics and bias** � BeerAdvocate skews toward craft beer enthusiasts and English-language reviewers. Food-101 skews toward Western dishes. Pairing table is curated by one perspective (Brewers Association). Joint model inherits all three biases. Mitigation ideas.
3. **Team contribution table** � each member's section ownership.

---

### STEP 19 � Streamlit deployment

**`deployment/app.py`**

Two-column layout:

- **Left:** Upload food photo ? CNN food label + confidence bar + Grad-CAM heatmap.
- **Right:** Three beer style cards, each showing: style name, joint compatibility score, top 3 flavor sentences from BiLSTM-ranked BeerAdvocate reviews.

Test locally: `streamlit run deployment/app.py`
Deploy: push to GitHub ? connect to Streamlit Cloud ? submit link.

---

## Final checklist before submission

- [ ] Notebook runs end to end without errors (Restart Kernel ? Run All)
- [ ] All 15 sections have visible outputs
- [ ] `figures/` contains all saved plots
- [ ] `weights/` contains all four `.pt` files
- [ ] `recommend()` runs on a new food photo end to end
- [ ] Joint model "unexpected pairings" experiment has visible output
- [ ] Streamlit app runs locally
- [ ] Deployment link works
- [ ] Notebook exported as PDF
- [ ] Presentation covers: business problem, EDA, CNN results, BiLSTM results, joint model + unexpected pairings, Grad-CAM, attention, deployment demo, ethics

---

Total steps: 19 | Phases: 3 | GPU required: Phase 2 only

| Phase | Where | Sections | Needs GPU |
| --- | --- | --- | --- |
| 1 | VS Code, CPU | 1�4, 6, 10�11, 13 (skeleton) | No |
| 2 | Google Colab | 5, 7�8, 13 (full) | **Yes** |
| 3 | VS Code, CPU | 9, 12, 14�15, deployment | No |

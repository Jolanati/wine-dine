# Qualio — Implementation Plan

## Step-by-step reference for beginners

---

## Before you start — one-time setup

1. Install Python 3.10+ and VS Code
2. Install the Jupyter extension in VS Code
3. Create the folder structure from `Work_idea.md`
4. Create `qualio.ipynb` inside the `qualio/` folder
5. Install packages in Cell 1 of the notebook:

   ```bash
   pip install torch torchvision datasets transformers
   pip install pandas numpy matplotlib seaborn scikit-learn
   pip install torchcam streamlit
   ```

---

## PHASE 1 — Local skeleton (VS Code, CPU, small data)

> Goal: every cell runs without errors. Accuracy does not matter yet.

---

### STEP 1 — Load the dataset (sanity check first)

- Download the WineSensed `small` split from HuggingFace
- **Print how many rows and how many MB** before doing anything else
- Save three CSV files to `data/`:
  - `images_reviews_attributes.csv` — vintages with labels, prices, ratings
  - `napping.csv` — the 2D taste coordinates
- Print the first 5 rows of each file so you can see the column names

**You are done when:** data loads without error and you can see column names.

---

### STEP 2 — Image EDA (exploratory data analysis)

- Load 16 wine label images and display them in a grid
- Print image dimensions (width × height × channels)
- Count how many images per style class (red / white / rosé / sparkling)
- Plot a bar chart of class distribution
- Check if classes are balanced — note it in a markdown cell

**You are done when:** you have a visible image grid and a bar chart.

---

### STEP 3 — Text EDA

- Load the Vivino reviews
- Print: total number of reviews, average word count per review
- Plot a histogram of review lengths
- Create 3 quality tier labels from star ratings:
  - 1–2 stars → Entry
  - 3 stars → Premium
  - 4–5 stars → Exceptional
- Plot a bar chart of tier distribution
- Show 3 example reviews from each tier

**You are done when:** you can see the tier bar chart and example reviews.

---

### STEP 4 — Image preprocessing

- Resize all images to 224×224
- Apply ImageNet normalisation (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Apply training augmentations: random horizontal flip, random rotation ±15°, colour jitter
- Split into train (70%) / val (15%) / test (15%) — stratified by style class, fixed random seed
- Create PyTorch `DataLoader` objects for each split

**You are done when:** you can load one batch and print its shape (e.g. `torch.Size([32, 3, 224, 224])`).

---

### STEP 5 — Text preprocessing

- Tokenise reviews at word level
- Load GloVe-100d embeddings — build a vocabulary from the training set only
- Pad or truncate sequences to a fixed max length (start with 100 tokens)
- Print: vocabulary size, % of reviews truncated
- Split into train / val / test (same 70/15/15, stratified by tier)
- Create PyTorch `DataLoader` objects

**You are done when:** you can load one batch and print its shape (e.g. `torch.Size([32, 100])`).

---

### STEP 6 — CNN from scratch (simple version)

- Build a small CNN with 3 convolutional blocks (Conv → ReLU → MaxPool)
- Add a fully connected head with 4 output neurons (one per style class)
- Train for 2 epochs on CPU
- Plot training loss and validation loss per epoch
- Print final validation accuracy

**You are done when:** loss goes down over 2 epochs (even slightly).

---

### STEP 7 — CNN with ResNet-18 (transfer learning)

- Load ResNet-18 with pre-trained ImageNet weights
- Freeze all layers except the last fully connected layer
- Replace the FC layer with a new one: 512 → 4 outputs
- Train for 2 epochs on CPU
- Plot training and validation loss/accuracy
- Compare accuracy with the from-scratch CNN in a markdown cell

**You are done when:** ResNet-18 trains and you have a comparison note.

---

### STEP 8 — LSTM (unidirectional, baseline)

- Build: Embedding layer (GloVe weights) → LSTM (128 hidden units) → Linear → 3 outputs
- Train for 2 epochs on CPU
- Plot training and validation loss/accuracy

**You are done when:** loss goes down.

---

### STEP 9 — BiLSTM (bidirectional, improved)

- Same as Step 8 but set `bidirectional=True` in the LSTM layer
- Add an attention mechanism (a simple weighted sum over hidden states)
- Train for 2 epochs on CPU
- Compare accuracy with the unidirectional LSTM in a markdown cell

**You are done when:** you have both LSTM variants trained and compared.

---

## PHASE 2 — Real training (Google Colab, GPU)

> Goal: get good weights. This is the only phase that needs a GPU.

---

### STEP 10 — Move to Colab

- Upload `qualio.ipynb` to Google Colab
- Mount Google Drive
- Change the dataset split from `small` to `full`
- Change training epochs: CNN → 10 epochs, BiLSTM → 5 epochs

---

### STEP 11 — Train CNN properly

- Train both CNN variants (from scratch and ResNet-18) for real
- Save the best checkpoint based on validation accuracy:
  - `weights/cnn_scratch.pt`
  - `weights/cnn_resnet18.pt`
- Report final test metrics: Accuracy, F1-score, Confusion Matrix

---

### STEP 12 — Run BiLSTM batch inference

- Train both LSTM variants for real
- Save `weights/bilstm.pt`
- Run the trained BiLSTM on **all vintages** in the dataset
- Save the predictions to `data/enriched_dataset.csv` (vintage ID + quality tier)
- Report final test metrics: Accuracy, F1-score, Confusion Matrix

---

### STEP 13 — Download artefacts back to local

Download from Colab/Drive to your `qualio/` folder:

- `weights/cnn_resnet18.pt`
- `weights/bilstm.pt`
- `data/enriched_dataset.csv`

---

## PHASE 3 — Analysis, explainability, deployment (VS Code, CPU)

> Goal: everything required by the rubric that doesn't need a GPU.

---

### STEP 14 — Evaluate final models (load saved weights)

- Load `cnn_resnet18.pt` and run it on the test set
- Load `bilstm.pt` and run it on the text test set
- Print and plot: Accuracy, F1-score, Confusion Matrix for both models
- Add a CSV guard: `if not os.path.exists('data/enriched_dataset.csv'): run batch inference`

---

### STEP 15 — CNN explainability (Grad-CAM)

- Pick 6 test images (2 correct predictions, 2 wrong, 2 uncertain)
- Use `torchcam` to generate a Grad-CAM heatmap for each
- Show the original image side by side with the heatmap
- Write a 3-sentence markdown interpretation: what regions drove the prediction?

---

### STEP 16 — BiLSTM explainability (attention weights)

- Pick 6 reviews (2 from each tier)
- Extract the attention weights from the BiLSTM
- Visualise as a colour-highlighted sentence (darker = more attention)
- Write a 3-sentence markdown interpretation: which words mattered most?

---

### STEP 17 — Napping similarity (taste neighbours)

- Load `napping.csv` — each wine has an (x, y) coordinate
- Write a function: given a wine ID, compute Euclidean distance to all others, return top 3
- Test it on 5 wines and display results as a small table

---

### STEP 18 — Business integration

- Write a `recommend(wine_id)` function that:
  1. Gets CNN style prediction (with confidence %)
  2. Looks up quality tier from `enriched_dataset.csv`
  3. Looks up wine fact card (name, grape, region, price, alcohol, rating)
  4. Runs napping similarity → top 3 taste neighbours
  5. Returns a plain-language sourcing recommendation
- Build a table of 20 examples showing CNN prediction + quality tier + recommendation side by side
- Plot a bar chart comparing CNN accuracy, BiLSTM accuracy, and combined recommendation quality

---

### STEP 19 — Business framing (markdown cells)

Add three markdown cells:

1. **Business framing** — Who uses Qualio? (restaurant/hotel buyers). What decision does it support? (stocking). What is the cost of a wrong prediction? (Entry wine recommended as Exceptional = buyer overpays)
2. **Ethics and bias** — Training labels come from English Vivino reviews only (language bias). Label images skew toward European wines (geographic bias). Mitigation ideas.
3. **Team contribution table** — List each team member and what they built.

---

### STEP 20 — Streamlit deployment

Create `deployment/app.py`:

- Section 1: Upload a wine label image → CNN predicts style with confidence
- Section 2: Paste a review → BiLSTM predicts quality tier with confidence
- Section 3: Combined sourcing recommendation shown as a card

Test locally: `streamlit run deployment/app.py`
Deploy: push to GitHub → connect repo to Streamlit Cloud → share the link.

---

## Final checklist before submission

- [ ] Notebook runs end to end without errors (restart kernel → run all)
- [ ] All plots are visible in the notebook output
- [ ] `weights/` folder contains saved `.pt` files
- [ ] `data/enriched_dataset.csv` exists
- [ ] Streamlit app runs locally
- [ ] Deployment link works
- [ ] Export notebook as PDF (`File → Export → PDF`)
- [ ] Presentation slides cover all 12 sections from the rubric

---

Total steps: 20 | Phases: 3 | GPU required: Phase 2 only (Google Colab, free tier is enough)

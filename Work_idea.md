# Qualio — Wine Intelligence

## Advanced Machine Learning · Final Project

---

## Product description

Qualio is a wine sourcing intelligence tool built for buyers at restaurants, hotels, and online retailers. The sourcing process today is manual — cross-referencing supplier sheets, critic scores, and purchase history takes time and expertise that not every buyer has.

Qualio reduces that to a single gesture. The buyer photographs a wine label. That is the only input. Everything else is computed automatically.

The label is passed through a Convolutional Neural Network trained on the WineSensed dataset. The CNN classifies the wine's style (red, white, rosé, sparkling) from the visual cues in the label — colour palette, typography, and layout conventions. Style classification is chosen over country-of-origin classification because it maps to visually cleaner boundaries; country prediction from label aesthetics alone is a harder task and expected accuracy would be low (~50–60%). The business value lies in the pipeline, not the accuracy number.

The system then matches the label to a vintage in the WineSensed database and retrieves its structured attributes — the wine name, grape composition, region, vintage year, alcohol percentage, and price. It also retrieves the Vivino user reviews for that vintage and passes them through a Bidirectional LSTM trained to classify review language into three quality tiers: Entry, Premium, or Exceptional. The LSTM does not run live at request time — it runs in batch during preprocessing and its predictions are stored, so the API response is fast.

The final section of the output is what makes Qualio distinct from a standard classifier. Using coordinates from the WineSensed napping experiment — where 256 participants tasted wines blind and physically placed stickers on paper to indicate flavour similarity — the system computes Euclidean distances in that 2D taste space and surfaces the three wines that real humans placed closest to the analysed bottle. These are not algorithmic recommendations. They are distances derived from embodied human experience of flavour.

The combined output is presented as a structured sourcing brief: country of origin with confidence, quality tier with review distribution, a full wine fact card, a plain-language stocking recommendation, and three wines that taste like this one.

## What it does

| Output | Source |
| --- | --- |
| Wine style (red / white / rosé / sparkling) | CNN — reads the label image |
| Quality tier (Entry / Premium / Exceptional) | BiLSTM — reads Vivino user reviews |
| Wine fact card (name, grape, region, price, alcohol, rating) | Retrieved from WineSensed dataset |
| Wines that taste like this one | Napping experiment — 256 real human tasters |

## Dataset

**WineSensed** — Learning to Taste, NeurIPS 2023
`https://huggingface.co/datasets/Dakhoo/L2T-NeurIPS-2023`

- 897k wine label images
- 824k Vivino user reviews
- 350k unique vintages with region, grape, price, alcohol, rating
- Napping coordinates — pairwise taste similarity from 256 blind tasters

## Models

### CNN — ResNet-18 (transfer learning)

- Input: wine label image
- Task: classify wine style (red, white, rosé, sparkling) — 4 classes
- Also trained from scratch for comparison
- Explainability: **Grad-CAM** visualisation on misclassified examples (~20 lines, `torchcam`)

### BiLSTM — GloVe-100d embeddings

- Input: Vivino review text
- Task: classify quality tier derived from star rating
- Also trained as unidirectional LSTM for comparison
- Runs in batch preprocessing — results stored in `enriched_dataset.csv`
- Explainability: **attention weights** visualisation showing which words drove the tier prediction
- Safety guard: notebook checks `if not os.path.exists('data/enriched_dataset.csv')` and re-runs batch inference rather than crashing

## Integration

CNN country prediction + BiLSTM quality tier → combined sourcing recommendation.
Napping Euclidean distances → three wines that real people found most similar in taste.

## Submission deliverables

As required by the course rubric:

| # | What | Details |
| --- | --- | --- |
| 1 | `qualio.ipynb` | Single notebook, all code and outputs, runs end to end |
| 2 | `qualio.pdf` | Same notebook exported as PDF |
| 3 | Presentation | 15-20 min slide deck (.pptx or .pdf) |
| 4 | Deployment link | Qualio hosted on **Streamlit Cloud** (simpler than FastAPI — no HTML file needed) |

## File structure

```text
qualio/
├── qualio.ipynb          ← the submission — fully self-contained
├── README.md             ← this file
├── requirements.txt
├── data/
│   ├── images_reviews_attributes.csv
│   ├── napping.csv
│   └── enriched_dataset.csv    ← generated after BiLSTM batch run
├── weights/
│   ├── cnn_resnet18.pt         ← saved after CNN training
│   └── bilstm.pt               ← saved after BiLSTM training
└── deployment/
    └── app.py                  ← Streamlit app (single file, no HTML needed)
```

All functions are defined inside `qualio.ipynb`. No external helper files. The notebook runs end to end on any machine without additional dependencies beyond the pip installs in cell 1.

## Development workflow

**Phase 1 — VS Code (local)**
Write all notebook sections locally. Use the WineSensed `small` split. Start with a dataset size-check cell — count rows and print file sizes before any training so memory issues surface early. Train for 2-3 epochs on CPU — the goal is a working pipeline, not accuracy.

**Phase 2 — Google Colab (GPU)**
Copy `qualio.ipynb` to Colab. Switch to the full dataset, set image size to 224x224, train for real. Save weights to Google Drive. Download `cnn_resnet18.pt`, `bilstm.pt`, and `enriched_dataset.csv` back to local.

**Phase 3 — VS Code (local)**
Continue with saved weights. All lightweight, runs on CPU. This phase covers:

- Grad-CAM visualisation (CNN explainability)
- Attention weight visualisation (BiLSTM explainability)
- Business integration logic and 20-example comparison table
- Business framing cell (end user, false positive/negative cost analysis, workflow integration)
- Ethics and bias cell (geographic bias in label training data, English-only review bias)
- Team contribution table
- Streamlit deployment (`deployment/app.py`)

---

Dataset: CC BY-NC-ND 4.0 · WineSensed, Bender et al., NeurIPS 2023

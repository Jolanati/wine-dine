# Beer Peer

## Advanced Machine Learning · Final Project

---

## Product description

Beer Peer is a beer recommendation app built for anyone sitting down to a meal. The question "what beer goes with this?" is common but rarely answered well — most people default to habit. Beer Peer makes expert pairing knowledge available through a single photograph.

The interaction takes under 10 seconds: photograph your food, get back three recommended beer styles with flavor descriptions written in the words of real beer drinkers.

---

## How it works

Three layers in sequence — each independent, each testable on its own:

```text
PHOTO  →  CNN  →  food label  →  lookup table  →  beer styles  →  BiLSTM  →  flavor language
```

| Layer | What it does | ML involved |
| --- | --- | --- |
| CNN | Classifies the food photo into one of 101 categories | Yes — image classification |
| Lookup table | Maps the food category to 2-3 compatible beer macro-styles | No — curated CSV |
| BiLSTM | Retrieves and ranks flavor language per style from BeerAdvocate | Yes — text classification |

The lookup table is the bridge. It is a simple CSV with 101 rows, generated from food-beer pairing principles (Brewers Association guidelines). No ML required in the bridge — this is intentional and honest.

---

## Example output

| | |
| --- | --- |
| **Input** | Photo of pizza margherita |
| **CNN output** | Pizza (94% confidence) |
| **Paired styles** | IPA · APA · Lager |
| **IPA** | "Citrusy, hoppy, bitter finish that cuts through rich tomato sauce and cheese." |
| **APA** | "Balanced hop character with a clean malt backbone. Lifts the saltiness." |
| **Lager** | "Crisp and clean. Light bitterness that refreshes between bites." |

---

## Outputs

| Output | How it is produced |
| --- | --- |
| **Food category** | CNN classifies the photo (101 Food-101 classes) |
| **Beer style recommendations** (2-3) | Lookup table: `food_beer_pairing.csv` |
| **Flavor language per style** | BiLSTM-ranked BeerAdvocate reviews for that style |
| **Compatibility score** *(+10 bonus)* | Joint model: food image embedding + beer description embedding → compatible / not |

---

## Datasets

### Image dataset — Food-101

`load_dataset("ethz/food101")`

- **101,000 images** across 101 food categories (750 train + 250 test per class)
- Clean, pre-labeled, loads in one line — no preprocessing required
- License: Open (ETH Zurich research dataset)
- **CNN task:** 101-class food classification

### Text dataset — BeerAdvocate Reviews

`kaggle datasets download rdoume/beerreviews`

- **1.5 million reviews** from the BeerAdvocate community
- Columns: `beer_style`, `review_text`, `review_overall`, `review_aroma`, `review_taste`
- **BiLSTM task:** classify review text into 8 beer macro-style categories
- License: Publicly available via Kaggle

### Pairing table — `food_beer_pairing.csv`

- 101-row CSV: `food, beer_style_1, beer_style_2, beer_style_3`
- Generated via LLM + Brewers Association pairing guidelines
- Embedded directly in the notebook (self-sufficient, no file dependency)

---

## Beer macro-style grouping

BeerAdvocate has 100+ styles. Grouped into 8 classes for BiLSTM training:

| Macro-style | Includes |
| --- | --- |
| Lager | American Lager, Pilsner, Helles, Munich Lager |
| IPA | American IPA, Double IPA, Session IPA, Imperial IPA |
| Stout/Porter | Dry Stout, Imperial Stout, Porter, Milk Stout |
| Wheat | Hefeweizen, Witbier, American Wheat, Berliner Weisse |
| Sour/Farmhouse | Saison, Gose, Lambic, Gueuze, Flanders |
| Amber/Brown | Amber Ale, Brown Ale, Red Ale, Irish Red |
| Pale Ale | American Pale Ale, Blonde, Kölsch, Cream Ale |
| Specialty | Barleywine, Fruit Beer, Smoked, Spiced |

---

## Models

### CNN — food classification from photo

- **Input:** food photograph (224×224 RGB)
- **Task:** 101-class food classification
- **Architecture 1:** Custom CNN trained from scratch (≥3 conv blocks)
- **Architecture 2:** ResNet-50 or EfficientNet-B0, frozen backbone + fine-tuned head
- **Explainability:** Grad-CAM — which part of the food photo drove the prediction?

### BiLSTM — beer style classification from review text

- **Input:** BeerAdvocate review text
- **Task:** 8-class macro-style classification
- **Architecture 1:** Unidirectional LSTM baseline
- **Architecture 2:** Bidirectional LSTM with attention
- **Embeddings:** GloVe-100d pre-trained word vectors
- **Explainability:** attention weight visualisation — which words drove the style prediction?
- **Inference mode:** style-conditioned retrieval — surface top-rated review sentences per style

### Joint Model — food-beer compatibility *(+10 bonus points)*

- **Positive pairs:** (food image, beer style description) from lookup table — label `1`
- **Negative pairs:** random (food image, beer style description) not in lookup — label `0`
- **Architecture:** frozen CNN encoder (512-d) + frozen BiLSTM encoder (256-d) → concat (768-d) → FC → compatible/not
- Train only the FC head; encoders are frozen
- **Value:** generalises beyond the lookup table — can score food-beer pairs that weren't hand-coded

---

## Business integration

The recommendation card returned to the user:

1. **Food identified** — "Pizza" (CNN, 94% confidence) + Grad-CAM heatmap
2. **Paired styles** — IPA · APA · Lager (from lookup table)
3. **Flavor language** — top BiLSTM-ranked review sentences per style
4. **Joint model score** *(bonus)* — compatibility confidence for each pairing

**Target users:** anyone at a restaurant, cooking at home, or stocking beer for an event. Primary demo: restaurant staff recommending beer pairings to guests.

---

## Submission deliverables

| # | What | Details |
| --- | --- | --- |
| 1 | `beer_peer.ipynb` | Single notebook, all code and outputs, runs end to end |
| 2 | `beer_peer.pdf` | Same notebook exported as PDF |
| 3 | Presentation | 15–20 min slide deck (.pptx or .pdf) |
| 4 | Deployment link | Beer Peer hosted on **Streamlit Cloud** |

---

## File structure

```text
beer-peer/
├── beer_peer.ipynb             ← the submission — fully self-contained
├── requirements.txt
├── weights/
│   ├── cnn_scratch.pt
│   ├── cnn_resnet50.pt
│   ├── lstm.pt
│   ├── bilstm.pt
│   └── joint_model.pt
├── figures/                    ← all saved plots (created by notebook)
└── deployment/
    └── app.py                  ← Streamlit app (single file)
```

---

## Notebook structure

| Section | Content |
| --- | --- |
| 1 | Environment setup — dependencies and imports |
| 2 | Data loading — Food-101 + BeerAdvocate + pairing table |
| 3 | EDA — image dataset (class distribution, sample grid) |
| 4 | EDA — text dataset (review length, style distribution, word clouds) |
| 5 | Image preprocessing and data loaders |
| 6 | Text preprocessing and data loaders |
| 7 | CNN — custom architecture (trained from scratch) |
| 8 | CNN — ResNet-50 (transfer learning) |
| 9 | CNN explainability — Grad-CAM |
| 10 | LSTM — unidirectional baseline |
| 11 | BiLSTM — bidirectional with attention |
| 12 | BiLSTM explainability — attention weights |
| 13 | Joint model — food-beer compatibility classifier *(+10 bonus)* |
| 14 | Business integration — recommendation card, 20-example table |
| 15 | Business framing, ethics, and team contributions |

---

## Development workflow

**Phase 1 — VS Code (local, CPU)**
Sections 1–4, 6, 10–12. Data loading, EDA, text preprocessing, LSTM + BiLSTM training. All CPU-friendly.

**Phase 2 — Google Colab (GPU)**
Sections 5, 7–9, 13. Image data loaders, CNN training, joint model. Save weights to Drive.

**Phase 3 — VS Code (local, CPU)**
Load saved weights. Grad-CAM, attention, recommendation card, Streamlit app, PDF export.

---

## One-sentence pitch

> "Beer Peer turns a food photo into a beer recommendation — combining computer vision, natural language understanding, and 1.5 million real beer drinker reviews."

---

Dataset licenses: Food-101 (ETH Zurich, research use) · BeerAdvocate (Kaggle, public)

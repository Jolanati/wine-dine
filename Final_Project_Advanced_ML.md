# Advanced Machine Learning — FINAL GROUP PROJECT

## Image Analysis with CNNs and Text Analysis with RNNs

---

## 1. Project Overview

This capstone project challenges you to build a solution that applies deep learning to two distinct data types within the same business domain. You will develop two specialised models, namely, a **Convolutional Neural Network (CNN)** for analysing images and a **Recurrent Neural Network (RNN/LSTM)** for analysing text, and then combine their outputs at the business decision level to produce actionable insights.

For example, in an e-commerce setting, the CNN might classify product images by category or detect visual defects, while the RNN analyses customer reviews for sentiment or complaint topics. Neither model feeds into the other at the architecture level; they are separate models solving separate tasks. The integration happens at the business layer: their predictions are combined in a dashboard, decision rule, or report that supports a concrete business action (e.g., flagging products that have both visual quality issues and negative review sentiment for urgent review).

This structure mirrors how deep learning is typically deployed in industry: specialised models for specialised data types, integrated at the application level rather than fused into a single monolithic architecture.

You will work in teams of up to 5 students. All written deliverables (the Jupyter Notebook, its PDF export, and the presentation file) are due by the deadline announced in class. Each team will present its results during the final lecture. In case your team cannot participate, upload the link to the presentation recording within your final submission.

---

## 2. Learning Objectives

By completing this project you will demonstrate your ability to:

- Design and train a CNN for an image analysis task, applying transfer learning from a pre-trained architecture (e.g., ResNet, EfficientNet, VGG).
- Design and train an RNN/LSTM for a text analysis task, including tokenisation, embedding, and sequence modelling.
- Evaluate each model independently using appropriate metrics and interpret its predictions using explainability techniques.
- Frame both tasks within a shared business context and demonstrate how their combined outputs support a concrete business decision.
- Deploy the trained models as a simple interactive prototype using Streamlit, Gradio, Flask, or FastAPI.
- Present technical results clearly in both written (notebook) and oral (presentation) formats.

---

## 3. Team Composition

- **Team size:** 1–5 students per team.
- **Registration:** Each team must submit the list of members via Moodle.
- **Contribution log:** Include a contribution table in the final notebook indicating which team member was responsible for which component (CNN, RNN, deployment, business analysis, etc.).

---

## 4. Dataset Requirements

Choose a business domain where both image data and text data are naturally available. You will need two datasets (or two parts of one dataset) from the same domain — one for the CNN task and one for the RNN task. The two datasets do not need to share the same samples, but they must relate to the same business context.

### 4.1 Minimum Dataset Criteria

**Image dataset (for CNN):**

- At least 1,000 images across at least 3 classes (for classification) or with continuous labels (for regression).
- Images should be at least 64×64 pixels; ideally 224×224 or higher for transfer learning.

**Text dataset (for RNN):**

- At least 1,000 text samples (reviews, descriptions, support tickets, clinical notes, etc.).
- Text fields should average at least 15 words per sample to give the RNN meaningful input.

Neither dataset may be a toy dataset or one used in lecture examples.

### 4.2 Recommended Sources

- [Kaggle Datasets](https://kaggle.com/datasets)
- [Amazon Product Data](https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews) — product images + customer review text.
- [Yelp Open Dataset](https://yelp.com/dataset) — business photos + review text.
- [UCI Machine Learning Repository](https://archive.ics.uci.edu)
- Any domain-specific open-data portal (healthcare, retail, real estate, hospitality, etc.).

### 4.3 Example Project Ideas

| Domain | CNN Task (Images) | RNN Task (Text) | Business Integration |
| --- | --- | --- | --- |
| **E-commerce** | Product image categorisation or defect detection | Customer review sentiment analysis | Flag products with visual issues AND negative reviews for urgent quality review |
| **Real estate** | Property photo room-type classification | Listing description quality scoring | Identify listings with poor photos AND weak descriptions for agent follow-up |
| **Healthcare** | Medical image classification (X-ray, dermoscopy) | Clinical note key-phrase extraction | Prioritise cases where image findings AND note keywords both indicate urgency |
| **Hospitality** | Hotel/restaurant photo quality scoring | Guest review topic and sentiment analysis | Dashboard showing visual quality vs. guest satisfaction by property |
| **Fashion retail** | Clothing category and attribute recognition | Product description and review analysis | Match visual style attributes with textual trend signals for buying decisions |
| **Food delivery** | Food photo appetisingness scoring | Menu description and review scoring | Rank menu items by visual appeal combined with customer satisfaction |

---

## 5. Technical Requirements (Detailed)

Your Jupyter Notebook must follow the structure outlined below. Each subsection corresponds to a graded component.

### 5.1 Data Loading and Exploratory Data Analysis (EDA)

Perform EDA separately for each dataset:

**Image dataset:**

- Display a grid of 9–16 sample images with their labels.
- Report image dimensions, channel statistics, and class distribution.
- Visualise class balance with a bar chart.

**Text dataset:**

- Report number of samples, text length distribution (histogram), and class/target distribution.
- Show a word cloud or top-N most frequent terms per class.
- Display 3–5 representative text samples per class.

### 5.2 Data Preprocessing

**Image preprocessing:**

- Resize all images to a consistent size (e.g., 224×224 for transfer learning).
- Apply normalisation consistent with the chosen pre-trained model (e.g., ImageNet mean/std for ResNet).
- Apply data augmentation (random flips, rotations, colour jitter) to the training set. Explain your augmentation choices.

**Text preprocessing:**

- Tokenise text using an appropriate strategy (word-level, subword, or character-level). Justify your choice.
- Build or load a vocabulary. Handle out-of-vocabulary tokens.
- Pad or truncate sequences to a fixed maximum length. Report the chosen length and the percentage of texts truncated.
- Optionally use pre-trained embeddings (GloVe, Word2Vec) and explain the trade-offs.

### 5.3 Train–Validation–Test Split

For each dataset independently, split into **training (70%)**, **validation (15%)**, and **test (15%)** sets using a fixed random seed for reproducibility. Ensure the split is stratified by the target variable.

The test set must remain untouched until final evaluation. Use the validation set for all architecture decisions, hyperparameter tuning, and early stopping.

### 5.4 CNN Branch (Image Model)

You must implement and compare at least **two CNN approaches**:

| # | Approach | Details |
| --- | --- | --- |
| 1 | **CNN trained from scratch** | Design a custom CNN architecture (at least 3 convolutional blocks with pooling). Report the number of trainable parameters and training curves (loss and accuracy vs. epoch for both training and validation sets). |
| 2 | **Transfer learning** | Load a pre-trained model (e.g., ResNet-18/50, EfficientNet-B0, VGG-16). Freeze the backbone, replace the classification head, and fine-tune. Optionally unfreeze later layers and compare results. |

For both approaches, report:

- Training and validation loss and accuracy curves.
- Final test-set performance using **Accuracy**, **F1-Score**, and **Confusion Matrix**.
- A discussion of which approach performs better and why (model capacity, data size, domain similarity to ImageNet, etc.).

### 5.5 RNN/LSTM Branch (Text Model)

You must implement a recurrent model for the text task:

- Use an Embedding layer (pre-trained or learned) followed by one or more LSTM (or GRU) layers.
- Experiment with at least **two architectural variations** (e.g., single vs. bidirectional LSTM, single vs. stacked layers, with vs. without dropout).
- Report training and validation loss and accuracy curves for each variation.
- Report final test-set performance using **Accuracy**, **F1-Score**, and **Confusion Matrix**.

**Bonus (optional, up to +3 points):** Replace or supplement the LSTM with a Transformer-based encoder (e.g., DistilBERT) and compare performance.

### 5.6 Business Integration

This is the core integrative component of the project. You must demonstrate how the two models work together at the business level:

- Define a concrete business decision or action that depends on outputs from both models (e.g., "flag a product for review if the CNN detects a visual defect AND the RNN detects negative sentiment").
- Implement the integration logic: a decision rule, scoring function, or dashboard that takes predictions from both models and produces a combined recommendation.
- Evaluate the integrated system: show examples where the combined output gives a better or more complete business insight than either model alone.
- Present a summary table or visualisation showing at least 10–20 examples where both models' predictions are displayed side-by-side with the combined business recommendation.

**Required visualisation:** A comparison chart showing the performance of the CNN (on its task), the RNN (on its task), and the combined business accuracy/quality of the integrated system.

### 5.7 Deployment Prototype

Deploy both models as an interactive prototype. The user should be able to:

- Upload or select an image and receive the CNN's prediction with a confidence score.
- Enter or paste text and receive the RNN's prediction with a confidence score.
- See the combined business recommendation based on both predictions.

**Frameworks:** Streamlit or Gradio (recommended for simplicity), Flask or FastAPI with a minimal HTML frontend.

Include a screenshot or screen recording of the working prototype. If deployment is infeasible due to model size, provide a standalone inference script with clear instructions.

### 5.8 Business Framing

Throughout the notebook and especially in the conclusion, frame your work in business terms. For example:

- What business decision does this pipeline support? Who is the end user?
- What is the cost of a false positive vs. a false negative for each model in this context?
- How would this pipeline be integrated into an existing business workflow?
- What are the ethical considerations (bias in images or text, fairness across demographic groups)?

---

## 6. Bonus: Joint Model (+10 points)

For teams that want an additional challenge, you may build a **joint model** that combines the CNN and RNN at the model level rather than only at the business decision level. This means:

- Extract the feature vector from the CNN (e.g., the output of the global average pooling layer before the classification head).
- Extract the feature vector from the RNN (e.g., the final hidden state of the LSTM).
- Concatenate both feature vectors and pass them through one or more fully connected layers to produce a single prediction.
- This requires a dataset where each sample has both an image and a text field with a shared target label (e.g., product image + product review → predict return likelihood).

If you attempt the joint model, you must still complete the full pipeline described in Sections 5.4–5.8. The joint model is evaluated **in addition to** (not instead of) the two individual models.

Present a comparison showing the performance of: (a) CNN-only, (b) RNN-only, and (c) the joint model.

> **Note:** This bonus requires a specific dataset where each sample contains both an image and text with the same label. Not all business domains will have such data readily available. Discuss feasibility with the instructor before committing to this bonus.

---

## 7. Deliverables

Each team must submit the following via Moodle by the stated deadline:

| # | Deliverable | Details |
| --- | --- | --- |
| 1 | **Jupyter Notebook (.ipynb)** | Complete, well-commented notebook with all code, outputs, and markdown explanations. Must run end-to-end without errors (given the dataset). Include the dataset or a download link/script. Trained model weights should be saved and loadable. |
| 2 | **PDF of the Notebook** | The same Jupyter Notebook exported as PDF. All outputs, plots, and markdown cells must be visible. |
| 3 | **Presentation (.pptx or .pdf)** | A 15–20 minute presentation summarising your project. The contribution of each team member must be specified. |
| 4 | **Deployment artefact** | Link to a hosted prototype (e.g., Hugging Face Spaces, Streamlit Cloud). |

---

## 8. Presentation

**Recommended structure (15–20 minutes + 5 minutes Q&A):**

1. **Title Slide** — Project title, team name, member names, date.
2. **Business Problem & Motivation** — What business problem are you solving? Why does it need both image and text analysis?
3. **Dataset Overview & EDA Highlights** — Key stats for both datasets, sample images, sample texts, class balance (2–3 visualisations).
4. **Architecture Design** — Diagrams of the CNN and RNN architectures. A flowchart showing how their outputs feed into the business integration layer.
5. **Results: CNN** — From-scratch vs. transfer learning comparison, training curves, best test-set metrics.
6. **Results: RNN** — Architectural variations comparison, training curves, best test-set metrics.
7. **Business Integration** — How the two models' outputs are combined, examples of joint recommendations, comparison visualisation.
8. **Interpretability** — Grad-CAM examples for images, attention/SHAP highlights for text.
9. **Deployment Demo** — Live demo or screen recording of the prototype.
10. **Business Impact & Ethics** — How would this be used in practice? Ethical risks and mitigation.
11. **Conclusions & Lessons Learned** — Summary, surprises, what you would explore next.
12. **Team Contributions & References**

---

## 9. Originality Policy

All submitted work must be the original product of your team. You may use online resources, documentation, and AI coding assistants as learning aids, but you must understand and be able to explain every line of your code. Direct copying of entire notebooks from the internet or other teams is strictly prohibited.

Cite any external code snippets, tutorials, pre-trained model sources, or datasets you use in a **References** section at the end of your notebook.

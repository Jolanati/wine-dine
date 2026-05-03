import json, uuid

nb_path = r'c:\Users\jolanta.stutane\Desktop\RSU_AI\DL_Final\wine-dine\wine-dine.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def md_cell(source):
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8],
            "metadata": {}, "source": source}

def code_cell(source):
    return {"cell_type": "code", "execution_count": None, "id": uuid.uuid4().hex[:8],
            "metadata": {}, "outputs": [], "source": source}

new_cells = []

# ── Header ──────────────────────────────────────────────────────────────────
new_cells.append(md_cell(
"""---

## Section 12 — RNN / LSTM Branch (Text Model)

This section trains two text classification models on the **WineSensed** wine-review dataset to predict **grape variety** (15 classes) from tasting-note text.

### Sub-sections

| Sub-section | Content |
|---|---|
| **12.1** | Text `Dataset` class + `DataLoader` objects |
| **12.2** | Variation 1 — Unidirectional LSTM baseline |
| **12.3** | Shared training utilities (`train_text_epoch`, `eval_text`) |
| **12.4** | Training the LSTM baseline |
| **12.5** | LSTM test evaluation — accuracy, Macro F1, curves, confusion matrix |
| **12.6** | Variation 2 — Bidirectional LSTM + Bahdanau attention |
| **12.7** | Training BiLSTM + Attention |
| **12.8** | BiLSTM test evaluation — accuracy, Macro F1, curves, confusion matrix |
| **12.9** | Side-by-side comparison: LSTM vs BiLSTM + Attention |

### Design choices

| Choice | Rationale |
|---|---|
| GloVe 100-d frozen → fine-tuned | Wine descriptors (tannins, terroir, minerality) are in GloVe's Wikipedia corpus; fine-tuning adapts them to the domain |
| `pack_padded_sequence` | Ignores `<PAD>` positions during LSTM forward pass — correct gradient flow |
| Gradient clipping (max-norm 5) | Standard RNN stability measure |
| Weighted CrossEntropy | Corrects for class imbalance across 15 grape varieties (`CLASS_WEIGHTS` from Section 7.5) |
| Early stopping (patience = 4) | Prevents overfitting on the relatively small WineSensed corpus |

**Prerequisite cells:** 1.2 (imports), 4.1 (splits), 7.1–7.5 (text preprocessing — `X_train`, `VOCAB`, `embedding_matrix`, `CLASS_WEIGHTS`)"""
))

# ── 12.1 ─────────────────────────────────────────────────────────────────────
new_cells.append(code_cell(
"""# ── 12.1  Text Dataset and DataLoaders ───────────────────────────────────────
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class WineTextDataset(torch.utils.data.Dataset):
    \"\"\"Dataset for padded token-id sequences + sequence lengths + labels.\"\"\"
    def __init__(self, sequences, labels):
        self.X       = torch.tensor(sequences, dtype=torch.long)
        self.y       = torch.tensor(labels,    dtype=torch.long)
        self.lengths = (self.X != 0).sum(dim=1).clamp(min=1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.lengths[i], self.y[i]


TXT_BATCH = 64

txt_train_ds = WineTextDataset(X_train, lbl_train)
txt_val_ds   = WineTextDataset(X_val,   lbl_val)
txt_test_ds  = WineTextDataset(X_test,  lbl_test)

txt_train_loader = DataLoader(txt_train_ds, batch_size=TXT_BATCH, shuffle=True,
                               num_workers=0, pin_memory=False)
txt_val_loader   = DataLoader(txt_val_ds,   batch_size=TXT_BATCH, shuffle=False,
                               num_workers=0, pin_memory=False)
txt_test_loader  = DataLoader(txt_test_ds,  batch_size=TXT_BATCH, shuffle=False,
                               num_workers=0, pin_memory=False)

print(f"{'Split':<10} {'Samples':>8}  {'Batches':>8}")
print("-" * 30)
for _name, _ds, _ldr in [("train", txt_train_ds, txt_train_loader),
                          ("val",   txt_val_ds,   txt_val_loader),
                          ("test",  txt_test_ds,  txt_test_loader)]:
    print(f"{_name:<10} {len(_ds):>8,}  {len(_ldr):>8,}")

_xb_t, _lb_t, _yb_t = next(iter(txt_train_loader))
print(f"\\nBatch shapes:")
print(f"  sequences : {_xb_t.shape}   (B x MAX_SEQ_LEN={MAX_SEQ_LEN})")
print(f"  lengths   : {_lb_t.shape}   min={_lb_t.min()}, max={_lb_t.max()}")
print(f"  labels    : {_yb_t.shape}   classes 0-{_yb_t.max()}")
print(f"\\n✓ 12.1 — Text DataLoaders ready.")"""
))

# ── 12.2 markdown ─────────────────────────────────────────────────────────────
new_cells.append(md_cell(
"""### 12.2 — Variation 1: Unidirectional LSTM Baseline

**Architecture:**
```
Input (B × L)  →  Embedding (GloVe 100-d, fine-tuned)
               →  Dropout(0.4)
               →  LSTM (100-d → 256-d, 2 layers, unidirectional)
               →  Last hidden state of final layer  (B × 256)
               →  Dropout(0.4)
               →  Linear(256 → 15)
```

- Reads each wine review **left-to-right only**; final hidden state summarises the full sequence.
- `pack_padded_sequence` ensures `<PAD>` positions do not contribute to gradients.
- This is the **baseline** — deliberately simple so improvements from Variation 2 are clearly attributable to bidirectionality and attention."""
))

# ── 12.2 code ─────────────────────────────────────────────────────────────────
new_cells.append(code_cell(
"""# ── 12.2  Unidirectional LSTM Baseline ───────────────────────────────────────
class LSTMBaseline(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes,
                 n_layers=2, dropout=0.4, embedding_matrix=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(
                torch.tensor(embedding_matrix, dtype=torch.float32))
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers,
                            batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_dim, n_classes)

    def forward(self, x, lengths):
        emb    = self.drop(self.embedding(x))
        packed = pack_padded_sequence(emb, lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        out = self.drop(hidden[-1])
        return self.fc(out)

    def encode(self, x, lengths):
        emb    = self.embedding(x)
        packed = pack_padded_sequence(emb, lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return hidden[-1]


HIDDEN_DIM  = 256
N_LAYERS    = 2
DROPOUT_RNN = 0.4

lstm_model = LSTMBaseline(
    vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
    n_classes=len(GRAPE_CLASSES), n_layers=N_LAYERS, dropout=DROPOUT_RNN,
    embedding_matrix=embedding_matrix,
).to(DEVICE)

_total = sum(p.numel() for p in lstm_model.parameters())
print(f"LSTMBaseline — total params : {_total:,}")
_out_lstm = lstm_model(_xb_t.to(DEVICE), _lb_t.to(DEVICE))
assert _out_lstm.shape == (TXT_BATCH, len(GRAPE_CLASSES))
print(f"Forward pass  : OK → {_out_lstm.shape}")
print("✓ 12.2 — LSTMBaseline ready.")"""
))

# ── 12.3 ─────────────────────────────────────────────────────────────────────
new_cells.append(code_cell(
"""# ── 12.3  Shared text training utilities ─────────────────────────────────────
def train_text_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for seqs, lengths, labels in loader:
        seqs, lengths, labels = seqs.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(seqs, lengths)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += len(labels)
    return total_loss / n, correct / n

@torch.no_grad()
def eval_text(model, loader, criterion):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for seqs, lengths, labels in loader:
        seqs, lengths, labels = seqs.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)
        logits      = model(seqs, lengths)
        total_loss += criterion(logits, labels).item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += len(labels)
    return total_loss / n, correct / n

criterion_txt = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(DEVICE))
print("✓ 12.3 — training utilities ready.")
print(f"  criterion_txt : CrossEntropyLoss(weight=CLASS_WEIGHTS)  — {len(GRAPE_CLASSES)} classes")"""
))

# ── 12.4 markdown ─────────────────────────────────────────────────────────────
new_cells.append(md_cell(
"""### 12.4 — Training the LSTM Baseline

| Setting | Value |
|---|---|
| Optimiser | Adam, lr = 1e-3, weight_decay = 1e-5 |
| LR scheduler | ReduceLROnPlateau × 0.5, patience 2 |
| Max epochs | 30 |
| Early stopping | patience = 4 |
| Gradient clipping | max-norm 5 |

Best weights saved to `weights/lstm_best.pt`."""
))

# ── 12.4 code ─────────────────────────────────────────────────────────────────
new_cells.append(code_cell(
"""# ── 12.4  Train the LSTM Baseline ────────────────────────────────────────────
TXT_EPOCHS   = 30
TXT_PATIENCE = 4

opt_lstm   = Adam(lstm_model.parameters(), lr=1e-3, weight_decay=1e-5)
sched_lstm = ReduceLROnPlateau(opt_lstm, factor=0.5, patience=2, verbose=False)

history_lstm       = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val_loss_lstm = float("inf")
no_improve_lstm    = 0
best_ckpt_lstm     = WEIGHTS / "lstm_best.pt"

print(f"Training LSTMBaseline  (max {TXT_EPOCHS} epochs, patience={TXT_PATIENCE})")
print(f"{'Epoch':>5}  {'tr_loss':>8}  {'tr_acc':>7}  {'vl_loss':>8}  {'vl_acc':>7}  {'lr':>9}")
print("-" * 57)

for epoch in range(1, TXT_EPOCHS + 1):
    tr_loss, tr_acc = train_text_epoch(lstm_model, txt_train_loader, criterion_txt, opt_lstm)
    vl_loss, vl_acc = eval_text(lstm_model, txt_val_loader, criterion_txt)
    history_lstm["train_loss"].append(tr_loss)
    history_lstm["val_loss"].append(vl_loss)
    history_lstm["train_acc"].append(tr_acc)
    history_lstm["val_acc"].append(vl_acc)
    sched_lstm.step(vl_loss)
    lr_now = opt_lstm.param_groups[0]["lr"]
    if vl_loss < best_val_loss_lstm:
        best_val_loss_lstm = vl_loss
        no_improve_lstm    = 0
        torch.save(lstm_model.state_dict(), best_ckpt_lstm)
        marker = " ✓"
    else:
        no_improve_lstm += 1
        marker = ""
    print(f"{epoch:>5}  {tr_loss:>8.4f}  {tr_acc:>7.4f}  {vl_loss:>8.4f}  {vl_acc:>7.4f}  {lr_now:>9.2e}{marker}")
    if no_improve_lstm >= TXT_PATIENCE:
        print(f"\\nEarly stopping at epoch {epoch}.")
        break

print(f"\\nBest val loss : {best_val_loss_lstm:.4f}  →  {best_ckpt_lstm}")
print("✓ 12.4 — LSTM Baseline training complete.")"""
))

# ── 12.5 ─────────────────────────────────────────────────────────────────────
new_cells.append(code_cell(
"""# ── 12.5  LSTM Baseline — test evaluation, curves, confusion matrix ───────────
lstm_model.load_state_dict(torch.load(best_ckpt_lstm, map_location=DEVICE))
lstm_model.eval()

all_preds_lstm, all_labels_lstm = [], []
with torch.no_grad():
    for seqs, lengths, labels in txt_test_loader:
        logits = lstm_model(seqs.to(DEVICE), lengths.to(DEVICE))
        all_preds_lstm.extend(logits.argmax(1).cpu().tolist())
        all_labels_lstm.extend(labels.tolist())

lstm_test_acc = sum(p == l for p, l in zip(all_preds_lstm, all_labels_lstm)) / len(all_labels_lstm)
lstm_f1       = f1_score(all_labels_lstm, all_preds_lstm, average="macro")
print(f"LSTM Baseline — Test accuracy : {lstm_test_acc:.4f}  ({lstm_test_acc*100:.2f}%)")
print(f"                Macro F1      : {lstm_f1:.4f}")
print()
print(classification_report(all_labels_lstm, all_preds_lstm, target_names=GRAPE_CLASSES, digits=3))

_epochs_lstm = range(1, len(history_lstm["train_loss"]) + 1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(_epochs_lstm, history_lstm["train_loss"], label="Train", lw=1.8)
axes[0].plot(_epochs_lstm, history_lstm["val_loss"],   label="Val",   lw=1.8)
axes[0].set_title("LSTM Baseline — Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
axes[1].plot(_epochs_lstm, history_lstm["train_acc"], label="Train", lw=1.8)
axes[1].plot(_epochs_lstm, history_lstm["val_acc"],   label="Val",   lw=1.8)
axes[1].set_title("LSTM Baseline — Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()
plt.suptitle(f"12.5 — LSTM Baseline  |  Test Acc: {lstm_test_acc*100:.2f}%  Macro-F1: {lstm_f1:.4f}", fontsize=12)
plt.tight_layout()
plt.savefig(FIGURES / "12_lstm_curves.png", dpi=100, bbox_inches="tight")
plt.show()

lstm_cm = confusion_matrix(all_labels_lstm, all_preds_lstm)
fig, ax = plt.subplots(figsize=(13, 11))
sns.heatmap(lstm_cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=GRAPE_CLASSES, yticklabels=GRAPE_CLASSES, ax=ax, linewidths=0.3)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"LSTM Baseline — Confusion Matrix  (Test Acc: {lstm_test_acc*100:.2f}%)")
plt.xticks(rotation=45, ha="right", fontsize=8); plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig(FIGURES / "12_lstm_cm.png", dpi=100, bbox_inches="tight")
plt.show()
print("✓ 12.5 — LSTM Baseline test evaluation complete.")"""
))

# ── 12.6 markdown ─────────────────────────────────────────────────────────────
new_cells.append(md_cell(
"""### 12.6 — Variation 2: Bidirectional LSTM + Bahdanau Attention

**Architecture:**
```
Input (B × L)  →  Embedding (GloVe 100-d, fine-tuned)
               →  Dropout(0.4)
               →  BiLSTM (100-d → 256-d × 2, 2 layers)
               →  Bahdanau attention over all L hidden states  →  B × 512
               →  Dropout(0.4)
               →  Linear(512 → 15)
```

**Bahdanau (additive) attention:**

$$\\text{score}(h_t) = \\mathbf{v}^\\top \\tanh(\\mathbf{W} h_t) \\qquad
\\alpha_t = \\text{softmax}(\\text{score}) \\qquad
\\mathbf{c} = \\sum_t \\alpha_t h_t$$

Attention lets the model up-weight grape-discriminative descriptor words (e.g. *tannins* for Cabernet, *tropical fruit* for Viognier) rather than compressing the whole sequence into a single vector."""
))

# ── 12.6 code ─────────────────────────────────────────────────────────────────
new_cells.append(code_cell(
"""# ── 12.6  Bidirectional LSTM + Bahdanau Attention ────────────────────────────
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.v    = nn.Linear(hidden_dim,     1,           bias=False)

    def forward(self, hidden_states, mask=None):
        energy  = torch.tanh(self.attn(hidden_states))
        scores  = self.v(energy).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        weights = torch.nan_to_num(weights, nan=0.0)
        context = (weights.unsqueeze(-1) * hidden_states).sum(dim=1)
        return context, weights


class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes,
                 n_layers=2, dropout=0.4, embedding_matrix=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(
                torch.tensor(embedding_matrix, dtype=torch.float32))
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.attention = BahdanauAttention(hidden_dim)
        self.drop      = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, x, lengths):
        emb    = self.drop(self.embedding(x))
        packed = pack_padded_sequence(emb, lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(output, batch_first=True, total_length=x.shape[1])
        mask    = (x != 0)
        context, _ = self.attention(output, mask)
        return self.fc(self.drop(context))

    def encode(self, x, lengths):
        emb    = self.embedding(x)
        packed = pack_padded_sequence(emb, lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(output, batch_first=True, total_length=x.shape[1])
        context, _ = self.attention(output, (x != 0))
        return context


bilstm_model = BiLSTMAttention(
    vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
    n_classes=len(GRAPE_CLASSES), n_layers=N_LAYERS, dropout=DROPOUT_RNN,
    embedding_matrix=embedding_matrix,
).to(DEVICE)

_total_bi = sum(p.numel() for p in bilstm_model.parameters())
print(f"BiLSTMAttention — total params : {_total_bi:,}")
_out_bi = bilstm_model(_xb_t.to(DEVICE), _lb_t.to(DEVICE))
assert _out_bi.shape == (TXT_BATCH, len(GRAPE_CLASSES))
print(f"Forward pass  : OK → {_out_bi.shape}")
print("✓ 12.6 — BiLSTMAttention ready.")"""
))

# ── 12.7 ─────────────────────────────────────────────────────────────────────
new_cells.append(code_cell(
"""# ── 12.7  Train BiLSTM + Attention ───────────────────────────────────────────
opt_bilstm   = Adam(bilstm_model.parameters(), lr=1e-3, weight_decay=1e-5)
sched_bilstm = ReduceLROnPlateau(opt_bilstm, factor=0.5, patience=2, verbose=False)

history_bilstm       = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val_loss_bilstm = float("inf")
no_improve_bilstm    = 0
best_ckpt_bilstm     = WEIGHTS / "bilstm_best.pt"

print(f"Training BiLSTMAttention  (max {TXT_EPOCHS} epochs, patience={TXT_PATIENCE})")
print(f"{'Epoch':>5}  {'tr_loss':>8}  {'tr_acc':>7}  {'vl_loss':>8}  {'vl_acc':>7}  {'lr':>9}")
print("-" * 57)

for epoch in range(1, TXT_EPOCHS + 1):
    tr_loss, tr_acc = train_text_epoch(bilstm_model, txt_train_loader, criterion_txt, opt_bilstm)
    vl_loss, vl_acc = eval_text(bilstm_model, txt_val_loader, criterion_txt)
    history_bilstm["train_loss"].append(tr_loss)
    history_bilstm["val_loss"].append(vl_loss)
    history_bilstm["train_acc"].append(tr_acc)
    history_bilstm["val_acc"].append(vl_acc)
    sched_bilstm.step(vl_loss)
    lr_now = opt_bilstm.param_groups[0]["lr"]
    if vl_loss < best_val_loss_bilstm:
        best_val_loss_bilstm = vl_loss
        no_improve_bilstm    = 0
        torch.save(bilstm_model.state_dict(), best_ckpt_bilstm)
        marker = " ✓"
    else:
        no_improve_bilstm += 1
        marker = ""
    print(f"{epoch:>5}  {tr_loss:>8.4f}  {tr_acc:>7.4f}  {vl_loss:>8.4f}  {vl_acc:>7.4f}  {lr_now:>9.2e}{marker}")
    if no_improve_bilstm >= TXT_PATIENCE:
        print(f"\\nEarly stopping at epoch {epoch}.")
        break

print(f"\\nBest val loss : {best_val_loss_bilstm:.4f}  →  {best_ckpt_bilstm}")
print("✓ 12.7 — BiLSTM + Attention training complete.")"""
))

# ── 12.8 ─────────────────────────────────────────────────────────────────────
new_cells.append(code_cell(
"""# ── 12.8  BiLSTM + Attention — test evaluation, curves, confusion matrix ──────
bilstm_model.load_state_dict(torch.load(best_ckpt_bilstm, map_location=DEVICE))
bilstm_model.eval()

all_preds_bilstm, all_labels_bilstm = [], []
with torch.no_grad():
    for seqs, lengths, labels in txt_test_loader:
        logits = bilstm_model(seqs.to(DEVICE), lengths.to(DEVICE))
        all_preds_bilstm.extend(logits.argmax(1).cpu().tolist())
        all_labels_bilstm.extend(labels.tolist())

bilstm_test_acc = sum(p == l for p, l in zip(all_preds_bilstm, all_labels_bilstm)) / len(all_labels_bilstm)
bilstm_f1       = f1_score(all_labels_bilstm, all_preds_bilstm, average="macro")
print(f"BiLSTM+Attention — Test accuracy : {bilstm_test_acc:.4f}  ({bilstm_test_acc*100:.2f}%)")
print(f"                   Macro F1      : {bilstm_f1:.4f}")
print()
print(classification_report(all_labels_bilstm, all_preds_bilstm, target_names=GRAPE_CLASSES, digits=3))

_epochs_bi = range(1, len(history_bilstm["train_loss"]) + 1)
fig, axes  = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(_epochs_bi, history_bilstm["train_loss"], label="Train", lw=1.8)
axes[0].plot(_epochs_bi, history_bilstm["val_loss"],   label="Val",   lw=1.8)
axes[0].set_title("BiLSTM + Attention — Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
axes[1].plot(_epochs_bi, history_bilstm["train_acc"], label="Train", lw=1.8)
axes[1].plot(_epochs_bi, history_bilstm["val_acc"],   label="Val",   lw=1.8)
axes[1].set_title("BiLSTM + Attention — Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()
plt.suptitle(f"12.8 — BiLSTM + Attention  |  Test Acc: {bilstm_test_acc*100:.2f}%  Macro-F1: {bilstm_f1:.4f}", fontsize=12)
plt.tight_layout()
plt.savefig(FIGURES / "12_bilstm_curves.png", dpi=100, bbox_inches="tight")
plt.show()

bilstm_cm = confusion_matrix(all_labels_bilstm, all_preds_bilstm)
fig, ax   = plt.subplots(figsize=(13, 11))
sns.heatmap(bilstm_cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=GRAPE_CLASSES, yticklabels=GRAPE_CLASSES, ax=ax, linewidths=0.3)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"BiLSTM + Attention — Confusion Matrix  (Test Acc: {bilstm_test_acc*100:.2f}%)")
plt.xticks(rotation=45, ha="right", fontsize=8); plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig(FIGURES / "12_bilstm_cm.png", dpi=100, bbox_inches="tight")
plt.show()
print("✓ 12.8 — BiLSTM + Attention test evaluation complete.")"""
))

# ── 12.9 markdown ─────────────────────────────────────────────────────────────
new_cells.append(md_cell(
"""### 12.9 — Comparison: LSTM Baseline vs. BiLSTM + Attention

Produces:
1. **4-panel training curves** — both models side-by-side
2. **Per-class accuracy bar chart** — which grape varieties each model handles better
3. **Summary table** — test accuracy, Macro F1, parameter count"""
))

# ── 12.9 code ─────────────────────────────────────────────────────────────────
new_cells.append(code_cell(
"""# ── 12.9  Side-by-side comparison ────────────────────────────────────────────
_lstm_params   = sum(p.numel() for p in lstm_model.parameters())
_bilstm_params = sum(p.numel() for p in bilstm_model.parameters())

print("=" * 65)
print(f"{'Model':<28} {'Test Acc':>10}  {'Macro F1':>10}  {'Params':>10}")
print("-" * 65)
print(f"{'LSTM Baseline':<28} {lstm_test_acc*100:>9.2f}%  {lstm_f1:>10.4f}  {_lstm_params:>10,}")
print(f"{'BiLSTM + Attention':<28} {bilstm_test_acc*100:>9.2f}%  {bilstm_f1:>10.4f}  {_bilstm_params:>10,}")
print("=" * 65)

e1 = range(1, len(history_lstm["train_loss"]) + 1)
e2 = range(1, len(history_bilstm["train_loss"]) + 1)
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes[0,0].plot(e1, history_lstm["train_loss"], label="Train", lw=1.8)
axes[0,0].plot(e1, history_lstm["val_loss"],   label="Val",   lw=1.8)
axes[0,0].set_title("LSTM Baseline — Loss"); axes[0,0].legend()
axes[0,1].plot(e1, history_lstm["train_acc"], label="Train", lw=1.8)
axes[0,1].plot(e1, history_lstm["val_acc"],   label="Val",   lw=1.8)
axes[0,1].set_title("LSTM Baseline — Accuracy"); axes[0,1].legend()
axes[1,0].plot(e2, history_bilstm["train_loss"], label="Train", lw=1.8)
axes[1,0].plot(e2, history_bilstm["val_loss"],   label="Val",   lw=1.8)
axes[1,0].set_title("BiLSTM + Attention — Loss"); axes[1,0].legend()
axes[1,1].plot(e2, history_bilstm["train_acc"], label="Train", lw=1.8)
axes[1,1].plot(e2, history_bilstm["val_acc"],   label="Val",   lw=1.8)
axes[1,1].set_title("BiLSTM + Attention — Accuracy"); axes[1,1].legend()
plt.suptitle("12.9 — LSTM vs BiLSTM + Attention: Training Curves", fontsize=13)
plt.tight_layout()
plt.savefig(FIGURES / "12_rnn_comparison.png", dpi=100, bbox_inches="tight")
plt.show()

def per_class_acc(preds, labels, n_classes):
    counts = [0]*n_classes; correct_c = [0]*n_classes
    for p, l in zip(preds, labels):
        counts[l]  += 1
        correct_c[l] += int(p == l)
    return [correct_c[i]/counts[i] if counts[i] else 0 for i in range(n_classes)]

lstm_pca   = per_class_acc(all_preds_lstm,   all_labels_lstm,   len(GRAPE_CLASSES))
bilstm_pca = per_class_acc(all_preds_bilstm, all_labels_bilstm, len(GRAPE_CLASSES))
x = range(len(GRAPE_CLASSES)); w = 0.38
fig, ax = plt.subplots(figsize=(16, 6))
ax.bar([i - w/2 for i in x], lstm_pca,   w, label="LSTM Baseline",     alpha=0.85, color="#4C72B0")
ax.bar([i + w/2 for i in x], bilstm_pca, w, label="BiLSTM + Attention", alpha=0.85, color="#DD8452")
ax.set_xticks(list(x)); ax.set_xticklabels(GRAPE_CLASSES, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Per-class Accuracy"); ax.set_ylim(0, 1.05)
ax.axhline(lstm_test_acc,   color="#4C72B0", lw=1.2, ls="--", alpha=0.6, label="LSTM overall")
ax.axhline(bilstm_test_acc, color="#DD8452", lw=1.2, ls="--", alpha=0.6, label="BiLSTM overall")
ax.set_title("12.9 — Per-class Accuracy: LSTM vs BiLSTM + Attention"); ax.legend()
plt.tight_layout()
plt.savefig(FIGURES / "12_rnn_per_class.png", dpi=100, bbox_inches="tight")
plt.show()

print("Figures: 12_rnn_comparison.png  12_rnn_per_class.png")
print("✓ Section 12 complete.  Next: Section 13 — Business Integration.")"""
))

# ── Append and save ──────────────────────────────────────────────────────────
nb['cells'].extend(new_cells)
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Done. Notebook now has {len(nb['cells'])} cells (+{len(new_cells)} added).")

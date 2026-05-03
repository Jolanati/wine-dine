import json, copy

nb_path = r'c:\Users\jolanta.stutane\Desktop\RSU_AI\DL_Final\wine-dine\wine-dine.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
print(f"Total cells before: {len(cells)}")

# ── Identify Section 11 cells (index 128=markdown header, 129=code) ──────────
# Cell 128 is the Section 11 markdown, cell 129 is the code cell
sec11_md   = cells[128]   # "## Section 11 — Save All Results..."
sec11_code = cells[129]   # the pickle/drive save code

# ── Remove them from their current position ───────────────────────────────────
cells.pop(129)  # code first (higher index)
cells.pop(128)  # then markdown

# ── Build a REPLACEMENT save cell that covers CNN + LSTM + BiLSTM ─────────────
new_save_markdown_src = (
    "---\n\n"
    "## Section 12 — Save All Results to Google Drive\n\n"
    "Bundles **all** results — CNN Scratch, ResNet-50, LSTM Baseline, and BiLSTM + Attention — "
    "into `results_snapshot.pkl` and syncs everything to Drive.\n\n"
    "| Dataset | Key | Weights file |\n"
    "|---|---|---|\n"
    "| CNN Scratch | `test_acc`, `macro_f1`, history, CM | `weights/cnn_scratch_best.pt` |\n"
    "| ResNet-50 | `rn_test_acc`, `rn_f1`, history, CM | `weights/cnn_resnet50_best.pt` |\n"
    "| LSTM Baseline | `lstm_test_acc`, `lstm_f1`, history, CM | `weights/lstm_best.pt` |\n"
    "| BiLSTM + Attention | `bilstm_test_acc`, `bilstm_f1`, history, CM | `weights/bilstm_best.pt` |\n\n"
    "Run this cell **after** Sections 8–11 have all completed.\n"
    "Reload in a future session with:\n"
    "```python\n"
    "import pickle\n"
    "snap = pickle.load(open('weights/results_snapshot.pkl', 'rb'))\n"
    "```"
)

new_save_code_src = (
    "# ── 12  Save ALL results to Google Drive ────────────────────────────────────\n"
    "import pickle, glob, shutil, os\n\n"
    "snapshot = {}\n\n"
    "# ── CNN Scratch ──────────────────────────────────────────────────────────────\n"
    "snapshot['cnn_scratch'] = {\n"
    "    'test_acc':         globals().get('test_acc'),\n"
    "    'macro_f1':         globals().get('macro_f1'),\n"
    "    'history':          globals().get('history_scratch'),\n"
    "    'all_preds':        globals().get('all_preds'),\n"
    "    'all_labels':       globals().get('all_labels'),\n"
    "    'confusion_matrix': globals().get('sc_cm'),\n"
    "    'per_class_acc':    globals().get('sc_per_class_acc'),\n"
    "    'weights_file':     'cnn_scratch_best.pt',\n"
    "}\n\n"
    "# ── ResNet-50 ─────────────────────────────────────────────────────────────────\n"
    "snapshot['resnet50'] = {\n"
    "    'test_acc':         globals().get('rn_test_acc'),\n"
    "    'macro_f1':         globals().get('rn_f1'),\n"
    "    'history':          globals().get('hist_rn'),\n"
    "    'all_preds':        globals().get('rn_preds'),\n"
    "    'all_labels':       globals().get('rn_labels'),\n"
    "    'confusion_matrix': globals().get('cm'),\n"
    "    'per_class_acc':    globals().get('per_class_acc'),\n"
    "    'weights_file':     'cnn_resnet50_best.pt',\n"
    "}\n\n"
    "# ── LSTM Baseline ─────────────────────────────────────────────────────────────\n"
    "snapshot['lstm'] = {\n"
    "    'test_acc':         globals().get('lstm_test_acc'),\n"
    "    'macro_f1':         globals().get('lstm_f1'),\n"
    "    'history':          globals().get('history_lstm'),\n"
    "    'all_preds':        globals().get('all_preds_lstm'),\n"
    "    'all_labels':       globals().get('all_labels_lstm'),\n"
    "    'confusion_matrix': globals().get('lstm_cm'),\n"
    "    'weights_file':     'lstm_best.pt',\n"
    "}\n\n"
    "# ── BiLSTM + Attention ────────────────────────────────────────────────────────\n"
    "snapshot['bilstm'] = {\n"
    "    'test_acc':         globals().get('bilstm_test_acc'),\n"
    "    'macro_f1':         globals().get('bilstm_f1'),\n"
    "    'history':          globals().get('history_bilstm'),\n"
    "    'all_preds':        globals().get('all_preds_bilstm'),\n"
    "    'all_labels':       globals().get('all_labels_bilstm'),\n"
    "    'confusion_matrix': globals().get('bilstm_cm'),\n"
    "    'weights_file':     'bilstm_best.pt',\n"
    "}\n\n"
    "# ── Text preprocessing artefacts ─────────────────────────────────────────────\n"
    "snapshot['text'] = {\n"
    "    'VOCAB':            globals().get('VOCAB'),\n"
    "    'VOCAB_SIZE':       globals().get('VOCAB_SIZE'),\n"
    "    'MAX_SEQ_LEN':      globals().get('MAX_SEQ_LEN'),\n"
    "    'embedding_matrix': globals().get('embedding_matrix'),\n"
    "    'GRAPE_CLASSES':    globals().get('GRAPE_CLASSES'),\n"
    "    'GRAPE_TO_IDX':     globals().get('GRAPE_TO_IDX'),\n"
    "    'CLASS_WEIGHTS':    globals().get('CLASS_WEIGHTS'),\n"
    "}\n\n"
    "# ── Split metadata ────────────────────────────────────────────────────────────\n"
    "snapshot['splits'] = {\n"
    "    'SEED':       SEED,\n"
    "    'train_size': globals().get('train_size'),\n"
    "    'test_size':  globals().get('test_size'),\n"
    "}\n\n"
    "# ── 1. Save snapshot locally ──────────────────────────────────────────────────\n"
    "snap_local = os.path.join(str(LOCAL_WEIGHTS), 'results_snapshot.pkl')\n"
    "with open(snap_local, 'wb') as f:\n"
    "    pickle.dump(snapshot, f)\n"
    "snap_size = os.path.getsize(snap_local) / 1e6\n"
    "print(f'✓ results_snapshot.pkl saved locally  ({snap_size:.1f} MB)')\n\n"
    "# ── 2. Copy snapshot to Drive ─────────────────────────────────────────────────\n"
    "if IN_COLAB:\n"
    "    snap_drive = os.path.join(WEIGHTS_DIR, 'results_snapshot.pkl')\n"
    "    shutil.copy2(snap_local, snap_drive)\n"
    "    print(f'✓ Snapshot → Drive: {snap_drive}')\n\n"
    "# ── 3. Copy weight files to Drive ────────────────────────────────────────────\n"
    "weight_files = [\n"
    "    'cnn_scratch_best.pt',\n"
    "    'cnn_resnet50_best.pt',\n"
    "    'lstm_best.pt',\n"
    "    'bilstm_best.pt',\n"
    "]\n"
    "if IN_COLAB:\n"
    "    print('\\nSyncing weight files to Drive ...')\n"
    "    for wf in weight_files:\n"
    "        src = os.path.join(str(LOCAL_WEIGHTS), wf)\n"
    "        dst = os.path.join(WEIGHTS_DIR, wf)\n"
    "        if os.path.exists(src):\n"
    "            shutil.copy2(src, dst)\n"
    "            size_mb = os.path.getsize(src) / 1e6\n"
    "            print(f'  ✓ {wf}  ({size_mb:.1f} MB)')\n"
    "        else:\n"
    "            print(f'  ✗ {wf}  NOT FOUND — run training cell first')\n"
    "else:\n"
    "    print('Weight files already on disk (local mode):')\n"
    "    for wf in weight_files:\n"
    "        src = os.path.join(str(LOCAL_WEIGHTS), wf)\n"
    "        exists = os.path.exists(src)\n"
    "        size_mb = os.path.getsize(src) / 1e6 if exists else 0\n"
    "        status = f'{size_mb:.1f} MB' if exists else 'MISSING'\n"
    "        print(f'  {\"✓\" if exists else \"✗\"}  {wf:<30} {status}')\n\n"
    "# ── 4. Bulk-sync figures to Drive ─────────────────────────────────────────────\n"
    "if IN_COLAB:\n"
    "    local_pngs = glob.glob(os.path.join(str(LOCAL_FIGURES), '*.png'))\n"
    "    synced, skipped = 0, 0\n"
    "    for src in sorted(local_pngs):\n"
    "        fname = os.path.basename(src)\n"
    "        dest  = os.path.join(FIGURES_DIR, fname)\n"
    "        if os.path.exists(dest) and os.path.getsize(dest) == os.path.getsize(src):\n"
    "            skipped += 1\n"
    "            continue\n"
    "        shutil.copy2(src, dest)\n"
    "        print(f'  ✓ Figure: {fname}')\n"
    "        synced += 1\n"
    "    print(f'Figures: {synced} synced, {skipped} already up-to-date.')\n"
    "else:\n"
    "    local_pngs = glob.glob(os.path.join(str(LOCAL_FIGURES), '*.png'))\n"
    "    print(f'\\nFigures on disk: {len(local_pngs)}')\n"
    "    for p in sorted(local_pngs):\n"
    "        print(f'  {os.path.basename(p)}')\n\n"
    "# ── 5. Summary table ─────────────────────────────────────────────────────────\n"
    "print('\\n' + '='*65)\n"
    "print(f'{\"Section\":<14} {\"Key\":<22} {\"Status\"}')\n"
    "print('-'*65)\n"
    "for section, data in snapshot.items():\n"
    "    for key, val in data.items():\n"
    "        if key == 'weights_file':\n"
    "            wf = os.path.join(str(LOCAL_WEIGHTS), val)\n"
    "            status = f'✓ saved ({os.path.getsize(wf)/1e6:.1f} MB)' if os.path.exists(wf) else '✗ MISSING'\n"
    "        else:\n"
    "            status = '✓' if val is not None else '✗  not in memory'\n"
    "        print(f'  {section:<14} {key:<22} {status}')\n"
    "print('='*65)\n"
    "print('\\n✓ Section 12 complete — all results saved.')\n"
    "print(f'  Snapshot: {snap_local}  ({snap_size:.1f} MB)')\n"
    "print('  Reload with:')\n"
    "print('    snap = pickle.load(open(\"weights/results_snapshot.pkl\", \"rb\"))')\n"
)

import uuid

new_md_cell = {
    "cell_type": "markdown",
    "id": uuid.uuid4().hex[:8],
    "metadata": {},
    "source": new_save_markdown_src
}
new_code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": uuid.uuid4().hex[:8],
    "metadata": {},
    "outputs": [],
    "source": new_save_code_src
}

# ── Append new save cells at the end ─────────────────────────────────────────
cells.append(new_md_cell)
cells.append(new_code_cell)

nb['cells'] = cells
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Done. Notebook now has {len(cells)} cells.")
print(f"  Old Section 11 save cells removed from position 128-129")
print(f"  New Section 12 save cells appended at end (cells {len(cells)-1} and {len(cells)})")

# ── Also update section12.ipynb ───────────────────────────────────────────────
s12_path = r'c:\Users\jolanta.stutane\Desktop\RSU_AI\DL_Final\section12.ipynb'
with open(s12_path, 'r', encoding='utf-8') as f:
    s12 = json.load(f)

s12['cells'].append(new_md_cell)
s12['cells'].append(new_code_cell)
with open(s12_path, 'w', encoding='utf-8') as f:
    json.dump(s12, f, ensure_ascii=False, indent=1)

print(f"\nsection12.ipynb updated to {len(s12['cells'])} cells (added save cells).")

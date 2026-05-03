"""
Restructure wine-dine.ipynb on disk:
  - Remove old Section 11 cells (positions 128-129)
  - Append new Section 13 (save all) at the end
"""
import json

path = r'c:\Users\jolanta.stutane\Desktop\RSU_AI\DL_Final\wine-dine\wine-dine.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
print(f'Before: {len(cells)} cells')
print(f'  Cell 128: {cells[128]["source"][0][:60].strip()}')
print(f'  Cell 129: {cells[129]["source"][0][:60].strip()}')

# Remove old Section 11 (indexes 128, 129)
del cells[129]  # delete code cell first (higher index)
del cells[128]  # delete markdown cell

print(f'After removal: {len(cells)} cells')

# New Section 13 markdown cell
sec13_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# 13 · Save ALL Results & Weights\n",
        "\n",
        "This cell saves **all four models'** metrics, predictions, confusion matrices, and weight files.  \n",
        "On Colab it also syncs everything to Google Drive.\n",
        "\n",
        "| Model | Weight file |\n",
        "|-------|------------|\n",
        "| CNN from scratch | `cnn_scratch_best.pt` |\n",
        "| ResNet-50 transfer | `cnn_resnet50_best.pt` |\n",
        "| LSTM baseline | `lstm_best.pt` |\n",
        "| BiLSTM + Attention | `bilstm_best.pt` |"
    ]
}

# New Section 13 code cell
sec13_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ── 13  Save ALL results ──────────────────────────────────────────────────────\n",
        "import pickle, glob, shutil, os\n",
        "\n",
        "snapshot = {}\n",
        "\n",
        "# CNN Scratch\n",
        "snapshot['cnn_scratch'] = {\n",
        "    'test_acc': globals().get('test_acc'),\n",
        "    'macro_f1': globals().get('macro_f1'),\n",
        "    'history': globals().get('history_scratch'),\n",
        "    'all_preds': globals().get('all_preds'),\n",
        "    'all_labels': globals().get('all_labels'),\n",
        "    'confusion_matrix': globals().get('sc_cm'),\n",
        "    'weights_file': 'cnn_scratch_best.pt',\n",
        "}\n",
        "\n",
        "# ResNet-50\n",
        "snapshot['resnet50'] = {\n",
        "    'test_acc': globals().get('rn_test_acc'),\n",
        "    'macro_f1': globals().get('rn_f1'),\n",
        "    'history': globals().get('hist_rn'),\n",
        "    'all_preds': globals().get('rn_preds'),\n",
        "    'all_labels': globals().get('rn_labels'),\n",
        "    'confusion_matrix': globals().get('cm'),\n",
        "    'weights_file': 'cnn_resnet50_best.pt',\n",
        "}\n",
        "\n",
        "# LSTM\n",
        "snapshot['lstm'] = {\n",
        "    'test_acc': globals().get('lstm_test_acc'),\n",
        "    'macro_f1': globals().get('lstm_f1'),\n",
        "    'history': globals().get('history_lstm'),\n",
        "    'all_preds': globals().get('all_preds_lstm'),\n",
        "    'all_labels': globals().get('all_labels_lstm'),\n",
        "    'confusion_matrix': globals().get('lstm_cm'),\n",
        "    'weights_file': 'lstm_best.pt',\n",
        "}\n",
        "\n",
        "# BiLSTM + Attention\n",
        "snapshot['bilstm'] = {\n",
        "    'test_acc': globals().get('bilstm_test_acc'),\n",
        "    'macro_f1': globals().get('bilstm_f1'),\n",
        "    'history': globals().get('history_bilstm'),\n",
        "    'all_preds': globals().get('all_preds_bilstm'),\n",
        "    'all_labels': globals().get('all_labels_bilstm'),\n",
        "    'confusion_matrix': globals().get('bilstm_cm'),\n",
        "    'weights_file': 'bilstm_best.pt',\n",
        "}\n",
        "\n",
        "# Text artefacts\n",
        "snapshot['text'] = {\n",
        "    'VOCAB': globals().get('VOCAB'),\n",
        "    'VOCAB_SIZE': globals().get('VOCAB_SIZE'),\n",
        "    'MAX_SEQ_LEN': globals().get('MAX_SEQ_LEN'),\n",
        "    'embedding_matrix': globals().get('embedding_matrix'),\n",
        "    'GRAPE_CLASSES': globals().get('GRAPE_CLASSES'),\n",
        "    'GRAPE_TO_IDX': globals().get('GRAPE_TO_IDX'),\n",
        "    'CLASS_WEIGHTS': globals().get('CLASS_WEIGHTS'),\n",
        "}\n",
        "\n",
        "snapshot['splits'] = {\n",
        "    'SEED': SEED,\n",
        "    'train_size': globals().get('train_size'),\n",
        "    'test_size': globals().get('test_size'),\n",
        "}\n",
        "\n",
        "# 1. Save snapshot locally\n",
        "snap_local = os.path.join(str(LOCAL_WEIGHTS), 'results_snapshot.pkl')\n",
        "with open(snap_local, 'wb') as f:\n",
        "    pickle.dump(snapshot, f)\n",
        "print(f'✓ results_snapshot.pkl saved  ({os.path.getsize(snap_local)/1e6:.1f} MB)')\n",
        "\n",
        "# 2. Copy snapshot to Drive\n",
        "if IN_COLAB:\n",
        "    shutil.copy2(snap_local, os.path.join(WEIGHTS_DIR, 'results_snapshot.pkl'))\n",
        "    print(f'✓ Snapshot → Drive: {WEIGHTS_DIR}')\n",
        "\n",
        "# 3. Sync weight files\n",
        "weight_files = ['cnn_scratch_best.pt', 'cnn_resnet50_best.pt',\n",
        "                'lstm_best.pt', 'bilstm_best.pt']\n",
        "print('\\nWeight files:')\n",
        "for wf in weight_files:\n",
        "    src = os.path.join(str(LOCAL_WEIGHTS), wf)\n",
        "    if os.path.exists(src):\n",
        "        mb = os.path.getsize(src) / 1e6\n",
        "        if IN_COLAB:\n",
        "            shutil.copy2(src, os.path.join(WEIGHTS_DIR, wf))\n",
        "            print(f'  ✓ {wf:<30} {mb:.1f} MB → Drive')\n",
        "        else:\n",
        "            print(f'  ✓ {wf:<30} {mb:.1f} MB (local)')\n",
        "    else:\n",
        "        print(f'  ✗ {wf:<30} MISSING')\n",
        "\n",
        "# 4. Sync figures\n",
        "local_pngs = sorted(glob.glob(os.path.join(str(LOCAL_FIGURES), '*.png')))\n",
        "if IN_COLAB:\n",
        "    synced = 0\n",
        "    for src in local_pngs:\n",
        "        dest = os.path.join(FIGURES_DIR, os.path.basename(src))\n",
        "        if not os.path.exists(dest) or os.path.getsize(dest) != os.path.getsize(src):\n",
        "            shutil.copy2(src, dest)\n",
        "            synced += 1\n",
        "    print(f'\\nFigures: {synced} synced to Drive, {len(local_pngs)-synced} already up-to-date.')\n",
        "else:\n",
        "    print(f'\\n{len(local_pngs)} figure(s) saved locally.')\n",
        "\n",
        "print('\\n✓ Section 13 complete — all results saved.')"
    ]
}

# Append new cells at the end
cells.append(sec13_md)
cells.append(sec13_code)

print(f'After append: {len(cells)} cells')
print(f'  Last cell: {cells[-1]["source"][0][:60].strip()}')

# Write back
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('\n✓ Done. Notebook saved to disk.')
print('  → In VS Code: Ctrl+Shift+P → "Revert File" to reload from disk.')

import json

wnd_path = r'c:\Users\jolanta.stutane\Desktop\RSU_AI\DL_Final\wine-dine\wine-dine.ipynb'
with open(wnd_path, 'r', encoding='utf-8') as f:
    wnd = json.load(f)

section12_cells = wnd['cells'][-14:]

nb = {
    'nbformat': 4, 'nbformat_minor': 5,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3 (ipykernel)', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.12.13'}
    },
    'cells': section12_cells
}

out_path = r'c:\Users\jolanta.stutane\Desktop\RSU_AI\DL_Final\section12.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'Done. section12.ipynb has {len(section12_cells)} cells')
for i, c in enumerate(section12_cells):
    src = c['source']
    preview = (src if isinstance(src, str) else ''.join(src))[:70].replace('\n', ' ')
    ct = c['cell_type']
    print(f'  {i+1}. [{ct}] {preview}')

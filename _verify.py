import json
with open(r'c:\Users\jolanta.stutane\Desktop\RSU_AI\DL_Final\wine-dine\wine-dine.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
cells = nb['cells']
for i in [127, 128, 129, 141, 142, 143]:
    c = cells[i]
    src = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
    ct = c['cell_type']
    preview = src[:90].replace('\n', ' ')
    print(f'Cell {i+1} [{ct}]: {preview}')

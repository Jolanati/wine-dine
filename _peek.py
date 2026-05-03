import json
with open(r'c:\Users\jolanta.stutane\Desktop\RSU_AI\DL_Final\wine-dine\wine-dine.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
for i in [129, 130]:
    c = nb['cells'][i]
    src = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
    ct = c['cell_type']
    print(f'--- Cell {i+1} [{ct}] ---')
    print(src)
    print()

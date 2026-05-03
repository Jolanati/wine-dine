import json

path = r'c:\Users\jolanta.stutane\Desktop\RSU_AI\DL_Final\wine-dine\wine-dine.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
print(f'Total cells: {len(cells)}')
print()

# Show cells around position 128-130 (0-indexed)
for i in range(127, min(len(cells), 135)):
    c = cells[i]
    src = c['source']
    first = (src[0][:80] if src else '(empty)').strip()
    print(f'  [{i}] {c["cell_type"]:<8} | {first}')

print()
# Show last 5 cells
print('Last 5 cells:')
for i in range(max(0, len(cells)-5), len(cells)):
    c = cells[i]
    src = c['source']
    first = (src[0][:80] if src else '(empty)').strip()
    print(f'  [{i}] {c["cell_type"]:<8} | {first}')

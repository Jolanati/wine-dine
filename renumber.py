"""Renumber sections: 12 → 11, 13 → 12 in wine-dine.ipynb"""
import json, re

path = r'c:\Users\jolanta.stutane\Desktop\RSU_AI\DL_Final\wine-dine\wine-dine.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# We only want to rename in cells 128+ (0-indexed) to avoid touching earlier content
changes = 0
for i in range(128, len(cells)):
    src = cells[i]['source']
    new_src = []
    for line in src:
        original = line
        # Replace "Section 12" → "Section 11"
        line = line.replace('Section 12', 'Section 11')
        # Replace "# 13 " → "# 12 " (markdown header)
        line = line.replace('# 13 ', '# 12 ')
        # Replace "── 13 " → "── 12 " (code comments)
        line = line.replace('── 13 ', '── 12 ')
        # Replace "12.1" → "11.1", "12.2" → "11.2", ... "12.9" → "11.9"
        for sub in range(1, 10):
            line = line.replace(f'12.{sub}', f'11.{sub}')
        # Replace "Section 13" → "Section 12" (if any)
        line = line.replace('Section 13', 'Section 12')
        line = line.replace('section 13', 'section 12')
        line = line.replace('section 12', 'section 11')  # lowercase too
        
        if line != original:
            changes += 1
        new_src.append(line)
    cells[i]['source'] = new_src

print(f'Made {changes} line replacements in cells 128+')

# Write back
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('✓ Saved. Sections renumbered: 12→11, 13→12')

import json
from collections import Counter
import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('microsoft/graphcodebert-base', local_files_only=True)
probs_m2 = np.load('results/m2_dr_5k_probs.npy')
th_m2 = 0.52

pairs = []
with open('results/bcb_5k_test_pairs_enriched.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            pairs.append(json.loads(line))

y_true = np.array([p['label'] for p in pairs])
preds = (probs_m2 >= th_m2).astype(int)
fn_idx = np.where((y_true == 1) & (preds == 0))[0]
fn_pairs = [pairs[i] for i in fn_idx]

tok_lens_1 = [len(tokenizer.tokenize(p['func1'])) for p in fn_pairs]
tok_lens_2 = [len(tokenizer.tokenize(p['func2'])) for p in fn_pairs]

def get_group(t):
    if t < 200:
        return 'Ngắn (< 200 tokens)'
    elif t <= 512:
        return 'Vừa (200 - 512 tokens)'
    else:
        return 'Dài (> 512 tokens)'

combos = Counter()
for t1, t2 in zip(tok_lens_1, tok_lens_2):
    g1 = get_group(t1)
    g2 = get_group(t2)
    combo = tuple(sorted([g1, g2]))
    combos[combo] += 1

print('=== THỐNG KÊ 125 CẶP FN THEO ĐỘ DÀI TOKEN ===')
total_asym = 0
for combo, cnt in combos.most_common():
    pct = (cnt / len(fn_pairs)) * 100
    if combo[0] != combo[1]:
        total_asym += cnt
    print(f'{combo[0]} -- {combo[1]}: {cnt} cặp ({pct:.1f}%)')

print(f'\nTổng số cặp lệch độ dài token: {total_asym}/{len(fn_pairs)} ({total_asym/len(fn_pairs)*100:.1f}%)')

# In chi tiết token cho 8 hàm top FN
top_ids = ['18891988', '2324868', '3584508', '8754809', '15241397', '14758866', '1180878', '10690321']
top_toks = {}
for p in fn_pairs:
    for id_key, func_key in [('id1', 'func1'), ('id2', 'func2')]:
        sid = p.get(id_key)
        if sid in top_ids and sid not in top_toks:
            top_toks[sid] = len(tokenizer.tokenize(p[func_key]))

print('\nToken của các hàm Hub trong FN:')
for sid in top_ids:
    print(f'  Hàm {sid}: {top_toks.get(sid)} tokens')

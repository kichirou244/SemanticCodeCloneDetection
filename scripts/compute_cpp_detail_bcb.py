import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from src.models.transformer_pair import TransformerPairClassifier
from src.models.astnn.model import ASTNNPairClassifier, to_device as astnn_to_device
from src.representations.ast.extract import compile_tree, extract_tree, collate_trees
from src.models.cpg_gnn.model import CPGGNNPairClassifier
from torch_geometric.data import Batch
from scripts.fast_native_cpg import FastNativeCPGExtractor

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = Path('data/custom_suite')
manifest = DATA_DIR / 'pairs.jsonl'
pairs = [json.loads(ln) for ln in manifest.read_text(encoding='utf-8').splitlines() if ln.strip()]
codes = {p.name: p.read_text(encoding='utf-8') for p in DATA_DIR.glob('*.cpp')}
for r in pairs:
    r['code_a'] = codes[r['a']]
    r['code_b'] = codes[r['b']]

def get_cat_acc(preds):
    cats = {}
    for p, pred in zip(pairs, preds):
        c = p.get('category')
        lbl = p.get('label')
        if c not in cats: cats[c] = {'tot': 0, 'cor': 0}
        cats[c]['tot'] += 1
        if pred == lbl: cats[c]['cor'] += 1
    return {k: f"{v['cor']/v['tot']*100:.1f}%" for k, v in cats.items()}

# 1. M1 BCB (th=0.88)
tok1 = AutoTokenizer.from_pretrained('microsoft/codebert-base')
enc1 = AutoModel.from_pretrained('microsoft/codebert-base')
m1 = TransformerPairClassifier('microsoft/codebert-base', encoder=enc1)
m1.load_state_dict(torch.load('results/exp_bcb_002/best_checkpoint.pt', map_location='cpu')['model_state'])
m1.to(DEVICE).eval()

probs1 = []
with torch.no_grad():
    for i in range(0, len(pairs), 16):
        t1 = tok1([p['code_a'] for p in pairs[i:i+16]], padding=True, truncation=True, max_length=256, return_tensors='pt').to(DEVICE)
        t2 = tok1([p['code_b'] for p in pairs[i:i+16]], padding=True, truncation=True, max_length=256, return_tensors='pt').to(DEVICE)
        p = torch.sigmoid(m1(t1, t2)).cpu().numpy()
        probs1.extend(p.tolist() if p.ndim > 0 else [float(p)])
preds1 = (np.array(probs1) >= 0.88).astype(int)
print("M1 BCB (th=0.88):", get_cat_acc(preds1))

# 2. M2 BCB (th=0.94)
tok2 = AutoTokenizer.from_pretrained('microsoft/graphcodebert-base')
enc2 = AutoModel.from_pretrained('microsoft/graphcodebert-base')
m2 = TransformerPairClassifier('microsoft/graphcodebert-base', encoder=enc2)
m2.load_state_dict(torch.load('results/exp_bcb_003/best_checkpoint.pt', map_location='cpu')['model_state'])
m2.to(DEVICE).eval()

probs2 = []
with torch.no_grad():
    for i in range(0, len(pairs), 16):
        t1 = tok2([p['code_a'] for p in pairs[i:i+16]], padding=True, truncation=True, max_length=256, return_tensors='pt').to(DEVICE)
        t2 = tok2([p['code_b'] for p in pairs[i:i+16]], padding=True, truncation=True, max_length=256, return_tensors='pt').to(DEVICE)
        p = torch.sigmoid(m2(t1, t2)).cpu().numpy()
        probs2.extend(p.tolist() if p.ndim > 0 else [float(p)])
preds2 = (np.array(probs2) >= 0.94).astype(int)
print("M2 BCB (th=0.94):", get_cat_acc(preds2))

# 3. M3 BCB (th=0.52)
m3 = ASTNNPairClassifier(hidden=200, dropout=0.2, n_stmt=45)
m3.load_state_dict(torch.load('results/exp_bcb_004/best_checkpoint.pt', map_location='cpu')['model_state'])
m3.to(DEVICE).eval()
cache3 = {p.name: compile_tree(extract_tree(p.read_text(encoding='utf-8'), language='cpp')) for p in DATA_DIR.glob('*.cpp')}
probs3 = []
with torch.no_grad():
    for i in range(0, len(pairs), 32):
        chunk = pairs[i:i+32]
        ba = astnn_to_device(collate_trees([cache3[p['a']] for p in chunk]), DEVICE)
        bb = astnn_to_device(collate_trees([cache3[p['b']] for p in chunk]), DEVICE)
        p = torch.sigmoid(m3(ba, bb)).cpu().numpy()
        probs3.extend(p.tolist() if p.ndim > 0 else [float(p)])
preds3 = (np.array(probs3) >= 0.52).astype(int)
print("M3 BCB (th=0.52):", get_cat_acc(preds3))

# 4. M4 BCB (th=0.70)
m4 = CPGGNNPairClassifier(hidden=128, num_layers=3, heads=4, dropout=0.2)
m4.load_state_dict(torch.load('results/exp_bcb_005/best_checkpoint.pt', map_location='cpu')['model_state'])
m4.to(DEVICE).eval()
ext4 = FastNativeCPGExtractor('cpp')
cache4 = {p.name: ext4.parse_to_cpg(p.read_text(encoding='utf-8')) for p in DATA_DIR.glob('*.cpp')}
probs4 = []
with torch.no_grad():
    for i in range(0, len(pairs), 32):
        chunk = pairs[i:i+32]
        ga = Batch.from_data_list([cache4[p['a']] for p in chunk]).to(DEVICE)
        gb = Batch.from_data_list([cache4[p['b']] for p in chunk]).to(DEVICE)
        p = torch.sigmoid(m4(ga, gb)).cpu().numpy()
        probs4.extend(p.tolist() if p.ndim > 0 else [float(p)])
preds4 = (np.array(probs4) >= 0.70).astype(int)
print("M4 BCB (th=0.70):", get_cat_acc(preds4))

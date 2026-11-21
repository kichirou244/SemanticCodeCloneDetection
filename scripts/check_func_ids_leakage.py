import json
import pandas as pd

target_ids = {'18891988', '2324868', '3584508', '8754809', '15241397', '14758866', '1180878', '10690321'}

print('Target IDs to check:', target_ids)

# 1. Kiểm tra trong train_shard_0.parquet
print('\n>>> Đang kiểm tra trong data/raw/bigclonebench/train_shard_0.parquet...')
df_tr = pd.read_parquet('data/raw/bigclonebench/train_shard_0.parquet', columns=['id1', 'id2', 'label'])
print(f'Tổng số cặp trong train_shard_0: {len(df_tr):,}')

# Kiểm tra tập con 30k train (15k pos + 15k neg)
pos_tr_30k = df_tr[df_tr['label'] == True].head(15000)
neg_tr_30k = df_tr[df_tr['label'] == False].head(15000)
df_tr_30k = pd.concat([pos_tr_30k, neg_tr_30k])

tr_funcs_all = set(df_tr['id1'].astype(str)).union(set(df_tr['id2'].astype(str)))
tr_funcs_30k = set(df_tr_30k['id1'].astype(str)).union(set(df_tr_30k['id2'].astype(str)))

# 2. Kiểm tra trong validation_shard_0.parquet
print('\n>>> Đang kiểm tra trong data/raw/bigclonebench/validation_shard_0.parquet...')
df_val = pd.read_parquet('data/raw/bigclonebench/validation_shard_0.parquet', columns=['id1', 'id2', 'label'])
print(f'Tổng số cặp trong val_shard_0: {len(df_val):,}')

pos_val_3k = df_val[df_val['label'] == True].head(1500)
neg_val_3k = df_val[df_val['label'] == False].head(1500)
df_val_3k = pd.concat([pos_val_3k, neg_val_3k])

val_funcs_all = set(df_val['id1'].astype(str)).union(set(df_val['id2'].astype(str)))
val_funcs_3k = set(df_val_3k['id1'].astype(str)).union(set(df_val_3k['id2'].astype(str)))

# 3. Kiểm tra các file jsonl trong results
def check_jsonl(path):
    s = set()
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if 'id1' in item and 'id2' in item:
                        s.add(str(item['id1']))
                        s.add(str(item['id2']))
    except Exception as e:
        print(f'Lỗi đọc {path}: {e}')
    return s

val_1k = check_jsonl('results/bcb_1k_val_pairs.jsonl')
tr_500 = check_jsonl('results/bcb_500_train_pairs.jsonl')

print('\n' + '=' * 80)
print(f'{"Mã hàm (ID)":<12} | {"Train (30k)":<12} | {"Train (Shard 0)":<16} | {"Val (3k)":<10} | {"Val (Shard 0)":<14} | {"Val (1k jsonl)":<14}')
print('-' * 80)

for fid in sorted(target_ids):
    in_tr_30k = 'CÓ' if fid in tr_funcs_30k else 'KHÔNG'
    in_tr_all = 'CÓ' if fid in tr_funcs_all else 'KHÔNG'
    in_val_3k = 'CÓ' if fid in val_funcs_3k else 'KHÔNG'
    in_val_all = 'CÓ' if fid in val_funcs_all else 'KHÔNG'
    in_val_1k = 'CÓ' if fid in val_1k else 'KHÔNG'
    print(f'{fid:<12} | {in_tr_30k:<12} | {in_tr_all:<16} | {in_val_3k:<10} | {in_val_all:<14} | {in_val_1k:<14}')

print('=' * 80)

# Đếm số lần xuất hiện cụ thể nếu có
print('\nChi tiết số lần xuất hiện (nếu có):')
for fid in sorted(target_ids):
    cnt_tr_30k = ((df_tr_30k['id1'].astype(str) == fid) | (df_tr_30k['id2'].astype(str) == fid)).sum()
    cnt_tr_all = ((df_tr['id1'].astype(str) == fid) | (df_tr['id2'].astype(str) == fid)).sum()
    cnt_val_3k = ((df_val_3k['id1'].astype(str) == fid) | (df_val_3k['id2'].astype(str) == fid)).sum()
    cnt_val_all = ((df_val['id1'].astype(str) == fid) | (df_val['id2'].astype(str) == fid)).sum()
    print(f'Hàm {fid}: Train_30k={cnt_tr_30k}, Train_Shard0={cnt_tr_all}, Val_3k={cnt_val_3k}, Val_Shard0={cnt_val_all}')

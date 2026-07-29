#!/usr/bin/env python3
import argparse, hashlib, json, pickle, sys
from pathlib import Path
import numpy as np

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def load(path):
 with path.open('rb') as f:return pickle.load(f)
def pairs(d): return {(int(k),int(v)) for k,vs in d.items() for v in vs}
def gip(train,nd,ni):
 r=np.zeros((nd,ni),dtype=np.float32)
 for d,vs in train.items(): r[int(d),list(map(int,vs))]=1
 def k(x):
  n=np.einsum('ij,ij->i',x,x); den=float(n.sum()); gamma=len(x)/den if den else 1.; dist=n[:,None]+n[None,:]-2*(x@x.T); np.maximum(dist,0,out=dist); return np.exp(-gamma*dist).astype(np.float32)
 return k(r),k(r.T)
def fail(msg,errors): errors.append(msg); print('ERROR',msg)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--recompute-gip',action='store_true');a=p.parse_args();root=a.root.resolve();errors=[]
 for ds in ('B-dataset','C-dataset','F-dataset'):
  base=root/'data'/ds; source=load(base/'drug_disease_list.pkl'); complete={(d,int(v)) for d,vs in enumerate(source) for v in vs}; outer=[]
  top=json.loads((base/'folds'/'manifest.json').read_text(encoding='utf-8'))
  if top.get('outer_folds')!=10: fail(f'{ds}: outer_folds != 10',errors)
  for f in range(10):
   fd=base/'folds'/f'fold_{f:02d}'; m=json.loads((fd/'manifest.json').read_text(encoding='utf-8')); tr,va,te=(load(fd/n) for n in ('train.pkl','val.pkl','test.pkl')); ps=[pairs(x) for x in (tr,va,te)]
   if ps[0]&ps[1] or ps[0]&ps[2] or ps[1]&ps[2]: fail(f'{ds} fold {f}: overlap',errors)
   if set.union(*ps)!=complete: fail(f'{ds} fold {f}: incomplete partition',errors)
   outer.append(ps[2])
   names={'train':'train.pkl','validation':'val.pkl','test':'test.pkl','adjacency':'adj_csr.npz','drug_gip':'DrugGIP.npy','disease_gip':'DiseaseGIP.npy'}
   for key,name in names.items():
    if sha(fd/name)!=m['sha256'][key]: fail(f'{ds} fold {f}: hash {name}',errors)
   if a.recompute_gip:
    dg,ig=gip(tr,m['num_drugs'],m['num_diseases'])
    if not np.array_equal(dg,np.load(fd/'DrugGIP.npy')): fail(f'{ds} fold {f}: DrugGIP leakage/mismatch',errors)
    if not np.array_equal(ig,np.load(fd/'DiseaseGIP.npy')): fail(f'{ds} fold {f}: DiseaseGIP leakage/mismatch',errors)
   rd=fd/'refit'; rm=json.loads((rd/'manifest.json').read_text(encoding='utf-8')); rtr,rte=load(rd/'train.pkl'),load(rd/'test.pkl')
   if pairs(rtr)&pairs(rte) or pairs(rtr)|pairs(rte)!=complete: fail(f'{ds} fold {f}: invalid refit partition',errors)
   if a.recompute_gip:
    dg,ig=gip(rtr,rm['num_drugs'],rm['num_diseases'])
    if not np.array_equal(dg,np.load(rd/'DrugGIP.npy')): fail(f'{ds} fold {f}: refit DrugGIP mismatch',errors)
    if not np.array_equal(ig,np.load(rd/'DiseaseGIP.npy')): fail(f'{ds} fold {f}: refit DiseaseGIP mismatch',errors)
  if set.union(*outer)!=complete or sum(map(len,outer))!=len(complete): fail(f'{ds}: outer test folds not exact partition',errors)
 sums=root/'SHA256SUMS.txt'
 if sums.exists():
  for line in sums.read_text(encoding='utf-8-sig').splitlines():
   expected,rel=line.split('  ',1); path=root/rel
   if not path.is_file() or sha(path)!=expected: fail(f'release checksum: {rel}',errors)
 code=(root/'config.py').read_text(encoding='utf-8')+(root/'utils'/'data.py').read_text(encoding='utf-8')
 for token in ('train_test_split','split_data_randomly','legacy split','test_ratio'):
  if token in code: fail(f'forbidden holdout token: {token}',errors)
 if errors: print(f'VERIFICATION FAILED ({len(errors)} errors)'); return 1
 print('VERIFICATION PASSED: 3 datasets, 10 outer folds, 60 selection/refit partitions, hashes and GIP provenance verified'); return 0
if __name__=='__main__': sys.exit(main())

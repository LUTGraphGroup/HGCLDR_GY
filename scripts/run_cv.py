#!/usr/bin/env python3
import argparse, subprocess, sys
from pathlib import Path
p=argparse.ArgumentParser(description='Run the corrected HGCLDR ten-fold protocol')
p.add_argument('--datasets',nargs='+',default=['B-dataset','C-dataset','F-dataset']); p.add_argument('--folds',nargs='+',type=int,default=list(range(10)))
p.add_argument('--device',default='cuda:0'); p.add_argument('--epochs',type=int,default=2000); p.add_argument('--eval-freq',type=int,default=10)
p.add_argument('--seed',type=int,default=1234); p.add_argument('--pseudo-mode',choices=['none','hard','weighted'],default='none'); p.add_argument('--run-tag',default='table3_main'); p.add_argument('--dry-run',action='store_true')
a=p.parse_args(); root=Path(__file__).resolve().parents[1]
for dataset in a.datasets:
  for fold in a.folds:
    if fold not in range(10): raise SystemExit(f'invalid fold {fold}')
    cmd=[sys.executable,'run.py','--dataset',dataset,'--fold',str(fold),'--device',a.device,'--epochs',str(a.epochs),'--eval-freq',str(a.eval_freq),'--seed',str(a.seed),'--refit_after_selection','1','--use_fixed_validation_pairs','1','--pseudo_mode',a.pseudo_mode,'--run_tag',a.run_tag]
    print(' '.join(cmd),flush=True)
    if not a.dry_run: subprocess.run(cmd,cwd=root,check=True)

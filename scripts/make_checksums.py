#!/usr/bin/env python3
import argparse, hashlib
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path('.')); a=p.parse_args(); root=a.root.resolve(); out=root/'SHA256SUMS.txt'
files=sorted(x for x in root.rglob('*') if x.is_file() and x != out and '__pycache__' not in x.parts and x.suffix != '.pyc')
out.write_text(''.join(f'{digest(x)}  {x.relative_to(root).as_posix()}\n' for x in files),encoding='utf-8')
print(f'wrote {out} ({len(files)} files)')

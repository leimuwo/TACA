#!/usr/bin/env python3
"""Download private ModelScope inference models and enforce the release SHA-256 lock."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def verify(folder, files):
    for name, expected in files.items():
        path = Path(folder) / name
        if not path.is_file() or path.stat().st_size != expected['bytes']:
            raise ValueError(f'Missing/size mismatch: {path}')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != expected['sha256']:
            raise ValueError(f'SHA-256 mismatch: {path}')


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=root / 'models')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    lock = json.loads((root / 'models.lock.json').read_text())
    for name, entry in lock['models'].items():
        dest = args.model_dir / name
        if not args.verify_only:
            from modelscope import snapshot_download
            snapshot_download(entry['repo_id'], revision=entry.get('revision', 'master'),
                              local_dir=str(dest), token=os.environ.get('MODELSCOPE_API_TOKEN'),
                              max_workers=2)
        verify(dest, entry['files'])
        print(f'{name}: verified {len(entry["files"])} inference files', flush=True)


if __name__ == '__main__':
    main()

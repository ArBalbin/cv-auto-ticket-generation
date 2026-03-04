#!/usr/bin/env python3
"""
scripts/create_data_yaml.py

Generates a data.yaml file for YOLOv8 training.

Usage:
    python scripts/create_data_yaml.py \
        --train ../data/merged/images/train \
        --val   ../data/merged/images/val \
        --names classes.txt \
        --out ../data/merged/data.yaml

If --names is not provided, it defaults to a single class ['person'].
The script validates the train/val folders and reports counts.
"""
import argparse
import os
import yaml
from pathlib import Path

def count_images(folder):
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    return sum(1 for p in Path(folder).rglob('*') if p.suffix.lower() in exts)

def read_names(names_path):
    with open(names_path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    return lines

def main():
    p = argparse.ArgumentParser(description="Create data.yaml for YOLO (ultralytics).")
    p.add_argument('--train', required=True, help='Path to training images folder (or images root).')
    p.add_argument('--val', required=True, help='Path to validation images folder (or images root).')
    p.add_argument('--names', required=False, help='Optional path to a text file with class names (one per line).')
    p.add_argument('--out', default=None, help='Output YAML path. Default: <train_parent>/data.yaml or provided path.')
    args = p.parse_args()

    train_path = Path(args.train).resolve()
    val_path = Path(args.val).resolve()

    if not train_path.exists():
        raise SystemExit(f"[ERROR] Train path does not exist: {train_path}")
    if not val_path.exists():
        raise SystemExit(f"[ERROR] Val path does not exist: {val_path}")

    # Determine class names
    if args.names:
        names_file = Path(args.names).resolve()
        if not names_file.exists():
            raise SystemExit(f"[ERROR] names file not found: {names_file}")
        names = read_names(names_file)
    else:
        names = ['person']  # default

    nc = len(names)

    # Output file path
    if args.out:
        out_path = Path(args.out).resolve()
    else:
        # default to parent of train folder
        out_path = train_path.parents[1] / 'data.yaml' if len(train_path.parents) >= 2 else train_path.parent / 'data.yaml'

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build yaml dict
    data = {
        'train': str(train_path),
        'val': str(val_path),
        'nc': nc,
        'names': names
    }

    # Write YAML
    with open(out_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False)

    # Print helpful information
    train_count = count_images(train_path)
    val_count = count_images(val_path)

    print(f"[OK] data.yaml written to: {out_path}")
    print(f"     train: {train_path}  ({train_count} images)")
    print(f"     val:   {val_path}  ({val_count} images)")
    print(f"     nc:    {nc}")
    print(f"     names: {names}")

if __name__ == '__main__':
    main()

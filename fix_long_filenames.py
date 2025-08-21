#!/usr/bin/env python3
# fix_long_filenames.py
# Usage examples:
#  python fix_long_filenames.py --dir uploads --dry-run
#  python fix_long_filenames.py --dir uploads
#  python fix_long_filenames.py --dir uploads --git

import argparse
import os
import sys
import unicodedata
import re
from pathlib import Path
import csv
import shutil
import subprocess

# ---------- config ----------
MAX_FILENAME_LEN = 100   # ตัวอักษรของชื่อไฟล์ (ไม่รวม path) ที่เราจะตัดเหลือ
ALLOWED_CHARS_RE = re.compile(r'[^A-Za-z0-9._-]')  # จะเปลี่ยน char ที่ไม่อนุญาตเป็น underscore
# ----------------------------

def to_ascii_slug(name: str, max_len: int = MAX_FILENAME_LEN) -> str:
    # แยกนามสกุล
    stem, dot, ext = name.rpartition('.')
    if dot == '':
        stem = name
        ext = ''
    # normalize unicode -> ascii approx
    nkfd = unicodedata.normalize('NFKD', stem)
    ascii_bytes = nkfd.encode('ascii', 'ignore')
    ascii_str = ascii_bytes.decode('ascii')
    # replace disallowed chars with underscore
    safe = ALLOWED_CHARS_RE.sub('_', ascii_str)
    # collapse multiple underscores
    safe = re.sub(r'__+', '_', safe).strip('_')
    if safe == '':
        safe = 'file'
    # truncate safely to keep room for suffixes
    if ext:
        max_stem = max_len - (len(ext) + 1)
    else:
        max_stem = max_len
    if len(safe) > max_stem:
        safe = safe[:max_stem]
    if ext:
        return f"{safe}.{ext}"
    else:
        return safe

def win_longpath(p: Path) -> str:
    # Return path string with \\?\ prefix on Windows to avoid 260 char limit.
    s = str(p.resolve())
    if os.name == 'nt':
        if s.startswith('\\\\?\\'):
            return s
        # UNC case
        if s.startswith('\\\\'):
            # UNC must become \\?\UNC\server\share\...
            return '\\\\?\\UNC\\' + s.lstrip('\\')
        return '\\\\?\\' + s
    return s

def safe_rename(src: Path, dst: Path, dry_run=False):
    """Perform rename/move, handling Windows longpath prefix if needed."""
    try:
        if dry_run:
            print(f"[DRY] rename: {src} -> {dst}")
            return True
        # Ensure destination directory exists
        dst.parent.mkdir(parents=True, exist_ok=True)
        if os.name == 'nt':
            os.rename(win_longpath(src), win_longpath(dst))
        else:
            src.rename(dst)
        print(f"[OK] rename: {src} -> {dst}")
        return True
    except Exception as e:
        print(f"[ERR] failed rename: {src} -> {dst}  ({e})")
        # Try fallback: copy+remove
        try:
            if dry_run:
                return False
            shutil.copy2(win_longpath(src) if os.name=='nt' else str(src), str(dst))
            src.unlink()
            print(f"[OK] copied+removed: {src} -> {dst}")
            return True
        except Exception as e2:
            print(f"[ERR] fallback also failed: {e2}")
            return False

def find_and_rename(root: Path, dry_run=False, do_git=False, map_csv='rename_map.csv'):
    root = root.resolve()
    if not root.exists():
        print("Error: directory not found:", root)
        return
    # We'll walk bottom-up (so files in deep folders processed first)
    rename_map = []  # tuples (old_rel, new_rel)
    seen_targets = set()
    for dirpath, dirs, files in os.walk(root, topdown=False):
        dirp = Path(dirpath)
        for f in files:
            src = dirp / f
            # skip .git and similar maybe
            if '.git' in src.parts:
                continue
            new_name = to_ascii_slug(f)
            # if new_name equals old but path too long, still may need to move to shorter path (rare)
            # build tentative dst path (keep same directory)
            dst = dirp / new_name
            # handle collisions: append counter
            counter = 1
            base, dot, ext = new_name.rpartition('.')
            while True:
                # if dst equals src (same file), break
                if src.samefile(dst) if dst.exists() and src.exists() else False:
                    # same file path
                    break
                # if dst already taken (by other file or planned rename), increment
                if dst.exists() or str(dst.resolve()) in seen_targets:
                    # make new candidate
                    if ext:
                        candidate = f"{base}_{counter}.{ext}"
                    else:
                        candidate = f"{base}_{counter}"
                    dst = dirp / candidate
                    counter += 1
                    continue
                break
            # if no change and not too long, skip
            old_rel = src.relative_to(root)
            new_rel = dst.relative_to(root)
            if str(old_rel) == str(new_rel):
                # still check path length; if path length > 250, we should attempt to shorten by moving into same dir with shorter name
                fullp = str(src.resolve())
                if os.name == 'nt' and len(fullp) > 250:
                    # try to shorten by truncating name
                    truncated = to_ascii_slug(f, max_len=50)
                    dst = dirp / truncated
                    new_rel = dst.relative_to(root)
                else:
                    # nothing to do
                    continue
            # perform rename
            ok = safe_rename(src, dst, dry_run=dry_run)
            if ok:
                rename_map.append((str(old_rel).replace('\\','/'), str(new_rel).replace('\\','/')))
                seen_targets.add(str(dst.resolve()))
                # if git requested, run git add for dst and git rm for src (but only after actual rename)
                if do_git and not dry_run:
                    try:
                        subprocess.run(['git', 'add', str(dst)], check=True)
                        # remove old path from index if existed
                        # git rm --cached -f oldpath may be needed if old path was tracked and name changed
                        # but safer to attempt git rm --cached and ignore errors
                        subprocess.run(['git', 'rm', '--cached', '-f', str(src)], check=False)
                    except Exception as e:
                        print(f"[WARN] git operation failed for {dst}: {e}")
            else:
                print(f"[WARN] skip on error: {src}")

    # write map csv
    if rename_map:
        with open(map_csv, 'w', newline='', encoding='utf-8') as csvf:
            writer = csv.writer(csvf)
            writer.writerow(['old', 'new'])
            for a,b in rename_map:
                writer.writerow([a,b])
        print(f"Wrote rename map to {map_csv} ({len(rename_map)} entries)")
    else:
        print("No files renamed.")

def main():
    p = argparse.ArgumentParser(description="Fix long/unfriendly filenames under a folder.")
    p.add_argument('--dir', '-d', default='uploads', help='root directory to scan (default: uploads)')
    p.add_argument('--dry-run', action='store_true', help='show actions but do not rename')
    p.add_argument('--git', action='store_true', help='run git add on new files (requires git in PATH)')
    p.add_argument('--map', default='rename_map.csv', help='CSV output mapping old->new (default rename_map.csv)')
    args = p.parse_args()
    root = Path(args.dir)
    find_and_rename(root, dry_run=args.dry_run, do_git=args.git, map_csv=args.map)

if __name__ == '__main__':
    main()

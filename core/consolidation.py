import json
import shutil
from pathlib import Path

import config
from core.database import DatabaseManager
from core.graph import GraphManager


def preview(db: DatabaseManager, output_dir: str) -> list[dict]:
    """
    Returns a list of move operations without executing them.
    Each entry: {parent, folder_id, folder_name, source, destination}
    """
    gm = GraphManager(db)
    groups = gm.get_consolidation_groups()
    folders = {f["folder_id"]: f for f in db.get_all_folders()}
    out = Path(output_dir)

    moves = []
    for seq, group in enumerate(groups, start=1):
        parent_name = f"{seq:03d}"
        for folder_id in group:
            folder = folders.get(folder_id)
            if not folder:
                continue
            src = Path(folder["original_path"])
            dst = out / parent_name / src.name
            moves.append(
                {
                    "parent": parent_name,
                    "folder_id": folder_id,
                    "folder_name": src.name,
                    "source": str(src),
                    "destination": str(dst),
                }
            )
    return moves


def check_preconditions(moves: list[dict]) -> list[str]:
    """Returns a list of error strings. Empty = safe to proceed."""
    errors = []
    destinations = set()

    for move in moves:
        dst = Path(move["destination"])
        dst_parent = dst.parent

        try:
            dst_parent.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            errors.append(f"No write permission: {dst_parent}")
            continue

        if str(dst) in destinations:
            errors.append(f"Duplicate destination: {dst}")
        destinations.add(str(dst))

    # Rough disk space check: sum source folder sizes
    total_bytes = 0
    for move in moves:
        src = Path(move["source"])
        if src.exists():
            total_bytes += sum(f.stat().st_size for f in src.rglob("*") if f.is_file())

    if moves:
        dst_drive = Path(moves[0]["destination"]).anchor
        try:
            free = shutil.disk_usage(dst_drive).free
            if free < total_bytes * 1.1:
                errors.append(
                    f"Insufficient disk space. Need ~{total_bytes // 1_048_576} MB, "
                    f"free {free // 1_048_576} MB."
                )
        except Exception:
            pass

    return errors


def commit(moves: list[dict]) -> Path:
    """
    Execute the moves. Writes an undo log before starting.
    Returns the path to the undo log.
    """
    undo_log = config.UNDO_LOG_PATH
    undo_entries = [{"from": m["destination"], "to": m["source"]} for m in moves]

    with open(undo_log, "w", encoding="utf-8") as f:
        json.dump(undo_entries, f, indent=2)

    for move in moves:
        src = Path(move["source"])
        dst = Path(move["destination"])

        if not src.exists():
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists():
            # Destination folder already exists — merge contents
            _merge_folders(src, dst)
        else:
            shutil.move(str(src), str(dst))

    return Path(undo_log)


def undo(log_path: Path):
    """Reverse the moves recorded in the undo log."""
    with open(log_path, encoding="utf-8") as f:
        entries = json.load(f)

    for entry in entries:
        src = Path(entry["from"])
        dst = Path(entry["to"])
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))

    log_path.unlink(missing_ok=True)


def _merge_folders(src: Path, dst: Path):
    """Move all contents of src into dst, renaming on collision."""
    for item in src.iterdir():
        target = dst / item.name
        if not target.exists():
            shutil.move(str(item), str(target))
        else:
            # Rename with incrementing suffix
            stem = item.stem if item.is_file() else item.name
            suffix = item.suffix if item.is_file() else ""
            counter = 1
            while True:
                new_name = f"{stem}_{counter}{suffix}"
                candidate = dst / new_name
                if not candidate.exists():
                    shutil.move(str(item), str(candidate))
                    break
                counter += 1
    if src.exists() and not any(src.iterdir()):
        src.rmdir()

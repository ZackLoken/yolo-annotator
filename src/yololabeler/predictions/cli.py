"""`yololabeler-import`: batch-import a prediction tree into an image tree,
headless.

Imports nothing from Tk or CustomTkinter, directly or transitively, so a batch
conversion runs on a machine with no display.
"""

from __future__ import annotations

import argparse
import os

from yololabeler.predictions.batch import batch_import, plan_pairs, totals
from yololabeler.predictions.importers import FORMATS


def _display(rel_path):
    """Relative path in forward slashes, as '(root)' for the roots."""
    return "(root)" if rel_path == "." else rel_path.replace(os.sep, "/")


def _is_within(child, parent):
    """True when child is parent or lives underneath it, case-insensitively on
    Windows.
    """
    child, parent = os.path.normcase(child), os.path.normcase(parent)
    try:
        return os.path.commonpath([child, parent]) == parent
    except ValueError:  # different drives
        return False


def _print_plan(plan):
    """Print the pairing table, both unmatched lists and any cross-folder stem
    collisions.
    """
    print(f"Source root: {plan.source_root}")
    print(f"Images root: {plan.images_root}")
    print(f"Format:      {plan.fmt}")
    print()
    print(f"Matched folder pairs ({len(plan.pairs)}):")
    width = max([len(_display(p.rel_path)) for p in plan.pairs] or [0])
    for pair in plan.pairs:
        print(
            f"  {_display(pair.rel_path):<{width}}  "
            f"{pair.source_files:>6} source files  {pair.images:>6} images"
        )
    if not plan.pairs:
        print("  (none)")
    for title, rel_paths in (
        (
            "Source folders with no image folder at the same relative path",
            plan.unmatched_sources,
        ),
        (
            "Image folders with no source folder at the same relative path",
            plan.unmatched_images,
        ),
    ):
        print()
        print(f"{title} ({len(rel_paths)}):")
        for rel_path in rel_paths:
            print(f"  {_display(rel_path)}")
        if not rel_paths:
            print("  (none)")
    print()
    print(
        f"Source stems found under more than one source folder "
        f"({len(plan.stem_collisions)}):"
    )
    for stem, rel_paths in plan.stem_collisions.items():
        print(f"  {stem}: {', '.join(_display(r) for r in rel_paths)}")
    if not plan.stem_collisions:
        print("  (none)")


def _build_parser():
    """Build the argument parser; dry run is the default and --write is the
    real run.
    """
    parser = argparse.ArgumentParser(
        prog="yololabeler-import",
        description="Batch-import a tree of prediction folders into the "
        "matching tree of image folders, pairing strictly by relative path. "
        "Previews the pairing without writing unless --write is given.",
    )
    parser.add_argument(
        "source_root", help="root of the tree of source prediction folders"
    )
    parser.add_argument(
        "images_root", help="root of the tree of image folders"
    )
    parser.add_argument(
        "--format",
        dest="fmt",
        required=True,
        choices=FORMATS,
        help="source prediction format",
    )
    parser.add_argument(
        "--model", required=True, help="model name recorded in each manifest"
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=None,
        help="class id for bur_detect_json, which carries none; "
        "ignored by the other formats",
    )
    parser.add_argument(
        "--user", default="", help="name recorded as imported_by"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        dest="write",
        action="store_false",
        default=False,
        help="print the pairing and write nothing (the default)",
    )
    mode.add_argument(
        "--write",
        dest="write",
        action="store_true",
        help="actually convert and write predictions/ under every image "
        "folder",
    )
    return parser


def main(argv=None):
    """Plan the pairing, print it, and import when --write was given. Returns
    an exit code.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    source_root = os.path.realpath(args.source_root)
    images_root = os.path.realpath(args.images_root)
    for root, label in (
        (source_root, "source root"),
        (images_root, "images root"),
    ):
        if not os.path.isdir(root):
            parser.error(f"{label} is not a directory: {root}")
    if source_root == images_root:
        parser.error(
            "source root and images root are the same folder; a batch job may "
            "not walk the tree it writes into"
        )
    if _is_within(source_root, images_root) or _is_within(
        images_root, source_root
    ):
        parser.error(
            "source root and images root are nested; a batch job may not walk "
            "the tree it writes into"
        )
    if args.fmt == "bur_detect_json" and args.class_id is None:
        parser.error(
            "bur_detect_json files carry no class; --class-id is required"
        )
    plan = plan_pairs(source_root, images_root, args.fmt)
    _print_plan(plan)
    print()
    if not plan.pairs:
        print("No folder pairs matched; nothing to import.")
        return 1
    if not args.write:
        print("Dry run: nothing written. Re-run with --write to import.")
        return 0
    results = batch_import(plan, args.model, args.class_id, args.user)
    for pair_result in results:
        print(
            f"{_display(pair_result.pair.rel_path)}: {pair_result.summary()}"
        )
    print()
    print(totals(results).summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

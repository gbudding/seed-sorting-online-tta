"""
parse_annotations.py
--------------------
Parses a GrainSet XML annotation file and returns a list of
(image_path, binary_label) pairs.

Real GrainSet XML format (confirmed from wheat_tiny.xml):

    <annotation>
      <object>
        <ID>Grainset_wheat_2021-10-18-12-30-42_7_p600s</ID>
        <species>wheat</species>
        <sub-species>hexaploid common</sub-species>
        <location>AU</location>
        <time>2021-06-19</time>
        <size>18</size>
        <DU_grain>NOR</DU_grain>
        <weight>47</weight>
      </object>
      ...
    </annotation>

Notes:
- Root tag    : <annotation>
- Per-kernel  : <object> child elements
- ID and DU_grain are child ELEMENTS with text content (not XML attributes)
- Image filename = <ID>.png
- Images live under: <data_root>/train/<category>/<ID>.png
                 or  <data_root>/test/<category>/<ID>.png
  where category = '0_NOR', '1_F&S', '2_SD', etc.

Binary label mapping:
  NOR -> 1  (good)
  everything else (F&S, SD, MY, AP, BN, BP, HD, UN, IM) -> 0  (bad)
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple


GOOD_CATEGORIES = {"NOR"}


def _get_text(element, tag: str) -> Optional[str]:
    """Safely get the text content of a child element."""
    child = element.find(tag)
    if child is None or child.text is None:
        return None
    return child.text.strip()


def find_image(image_id: str, data_root: Path) -> Optional[Path]:
    """
    Search for <image_id>.png anywhere under data_root.
    Checks train/<category>/ and test/<category>/ subdirectories.
    Returns first match found, or None.
    """
    for split in ["train", "test"]:
        split_dir = data_root / split
        if not split_dir.exists():
            continue
        for category_dir in split_dir.iterdir():
            if not category_dir.is_dir():
                continue
            candidate = category_dir / f"{image_id}.png"
            if candidate.exists():
                return candidate
    return None


def parse_xml(xml_path: str, data_root: str) -> List[Tuple[str, int]]:
    """
    Parse a GrainSet XML annotation file and return
    a list of (absolute_image_path, binary_label) tuples.

    Parameters
    ----------
    xml_path  : path to species XML, e.g. 'data/wheat/wheat.xml'
    data_root : directory containing train/ and test/ subfolders,
                e.g. 'data/wheat/'

    Returns
    -------
    List of (path_str, label) where label 1=good, 0=bad.
    """
    data_root = Path(data_root)
    tree      = ET.parse(xml_path)
    root      = tree.getroot()

    samples: List[Tuple[str, int]] = []
    missing = 0
    skipped = 0

    for obj in root.findall("object"):
        image_id = _get_text(obj, "ID")
        du_grain = _get_text(obj, "DU_grain")

        if image_id is None or du_grain is None:
            skipped += 1
            continue

        image_path = find_image(image_id, data_root)
        if image_path is None:
            missing += 1
            continue

        label = 1 if du_grain in GOOD_CATEGORIES else 0
        samples.append((str(image_path), label))

    if missing > 0:
        print(f"[parse_xml] Warning: {missing} images not found under {data_root}")
    if skipped > 0:
        print(f"[parse_xml] Warning: {skipped} objects skipped (missing ID or DU_grain)")

    return samples


def parse_from_folders(data_root: str) -> List[Tuple[str, int]]:
    """
    Alternative: derive labels purely from folder names without the XML.
    Walks <data_root>/train/<category>/ and test/<category>/.
    Folder prefix '0' = NOR = good (label 1), all others = bad (label 0).
    """
    data_root = Path(data_root)
    samples: List[Tuple[str, int]] = []

    for split in ["train", "test"]:
        split_dir = data_root / split
        if not split_dir.exists():
            continue
        for category_dir in split_dir.iterdir():
            if not category_dir.is_dir():
                continue
            # '0_NOR' -> good, '1_F&S', '2_SD', ... -> bad
            prefix = category_dir.name.split("_")[0]
            label  = 1 if prefix == "0" else 0
            for img in sorted(category_dir.glob("*.png")):
                samples.append((str(img), label))

    return samples


def class_distribution(samples: List[Tuple[str, int]]) -> dict:
    """Return counts and percentages for each class."""
    from collections import Counter
    counts = Counter(label for _, label in samples)
    total  = len(samples)
    return {
        "total"    : total,
        "good"     : counts[1],
        "bad"      : counts[0],
        "pct_good" : 100 * counts[1] / total if total else 0,
        "pct_bad"  : 100 * counts[0] / total if total else 0,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test XML parser on a GrainSet species.")
    parser.add_argument("xml",  help="Path to XML, e.g. data/wheat/wheat.xml")
    parser.add_argument("root", help="Image root, e.g. data/wheat")
    parser.add_argument("--use-folders", action="store_true",
                        help="Use folder-based parsing instead of XML")
    args = parser.parse_args()

    if args.use_folders:
        print("Using folder-based parsing ...")
        samples = parse_from_folders(args.root)
    else:
        print("Using XML parsing ...")
        samples = parse_xml(args.xml, args.root)

    dist = class_distribution(samples)
    print(f"\nParsed {dist['total']:,} samples")
    print(f"  Good (NOR) : {dist['good']:,}  ({dist['pct_good']:.1f}%)")
    print(f"  Bad (DU+IM): {dist['bad']:,}  ({dist['pct_bad']:.1f}%)")
    print(f"\nFirst 5 samples:")
    for path, lbl in samples[:5]:
        print(f"  [{lbl}] {path}")

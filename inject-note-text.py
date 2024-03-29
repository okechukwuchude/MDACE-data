import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict

# Setting up logging
logger = logging.getLogger(Path(__file__).name)


def inject_note_text(notes_map: Dict[int, str], admission: Dict) -> Dict:
    """Inject text in-place into admission"""
    for note in admission["notes"]:
        note_id = note.get("note_id")  # Get note_id from note
        if note_id is not None:
            text = notes_map.get(note_id)  # Get text from notes_map
            if text is not None:
                note["text"] = text
                for annotation in note.get("annotations", []):
                    begin = annotation.get("begin", 0)
                    end = annotation.get("end", 0)
                    annotation["covered_text"] = text[begin:end]
            else:
                logger.warning(f"No text found for note_id: {note_id}")
        else:
            logger.warning("No note_id found in note")

    return admission


def _make_out_path(json_file: Path, input_dir: Path, out_dir: Path) -> Path:
    """Generate output path for injected JSON file"""
    prefix_len = len(input_dir.parts)
    return out_dir.joinpath(*json_file.parts[prefix_len:])


def inject_and_persist(notes_map: Dict[int, str], data_dir: Path, out_dir: Path):
    """Inject text into admission and persist in out_dir if provided"""
    if out_dir:
        logger.info(f"Injecting text and persisting to {out_dir.absolute()}")
    else:
        logger.info("Injecting text in place")

    # Iterate through JSON files in data_dir
    for json_file in data_dir.glob("**/*.json"):
        with open(json_file, "r", encoding="utf8") as ifp:
            admission = inject_note_text(notes_map, json.load(ifp))

        # Create output path
        if out_dir:
            out_path = _make_out_path(json_file, data_dir, out_dir)
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_path = json_file

        # Write injected admission to output file
        with open(out_path, "w", encoding="utf8") as ofp:
            json.dump(admission, ofp, indent=2)


def build_notes_map(noteevents: Path) -> Dict[int, str]:
    """Build mapping from note_id to text from NOTEEVENTS.csv"""
    logger.info(f"Loading {noteevents}")
    id_text_map = dict()
    csv.field_size_limit(1024 * 1024 * 1024)
    with open(noteevents, "r", encoding="utf8") as ifp:
        reader = csv.reader(ifp)
        # Skip header
        next(reader)
        for row in reader:
            note_id, text = int(row[0]), row[10]
            id_text_map[note_id] = text
    return id_text_map


def main(args: argparse.Namespace):
    # Build mapping from note_id to text
    notes_map = build_notes_map(args.noteevents)
    # Inject text into admissions and persist if out_dir provided
    inject_and_persist(notes_map, args.data_dir, args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Define command-line arguments
    parser.add_argument(
        "--noteevents", help="Path to NOTEEVENTS.csv", type=Path, required=True
    )
    parser.add_argument(
        "--data-dir",
        help="Path to top level MDACE data directory",
        type=Path,
        default="data",
    )
    parser.add_argument(
        "--out-dir",
        help="Write JSON files with text injected into them",
        type=Path,
        default=None,
    )

    # Configure logging level
    logging.basicConfig(level=logging.INFO)
    # Parse command-line arguments and execute the main function
    main(parser.parse_args())

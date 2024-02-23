import argparse
import dataclasses
import logging
from pathlib import Path
from typing import List, Dict, Set

# Importing necessary modules from the mdace package
from mdace.cleanup import trim_annotations, merge_adjacent_annotations
from mdace.data import MDACEData, Admission
from mdace.metrics import AllErrorRates
from mdace.text import tokenize

# Setting up logging
logger = logging.getLogger(Path(__file__).name)

# Function to filter admissions by given admission IDs
def filter_admissions_by_id(dataset: MDACEData, hadm_ids: Set[int]) -> MDACEData:
    return MDACEData(
        admissions=[adm for adm in dataset.admissions if adm.hadm_id in hadm_ids]
    )

# Function to filter notes by given note categories
def filter_notes_by_category(dataset: MDACEData, note_categories: Set[str]) -> MDACEData:
    return MDACEData(
        admissions=[
            dataclasses.replace(
                adm,
                notes=[note for note in adm.notes if note.category in note_categories],
            )
            for adm in dataset.admissions
        ]
    )

# Function to load admission IDs from a split file
def load_hadm_ids(split_file: Path) -> Set[int]:
    hadm_ids = set()
    with open(split_file, "r") as ifp:
        for idx, line in enumerate(ifp):
            try:
                hadm_ids.add(int(line.strip()))
            except ValueError as ve:
                if idx > 0:
                    # If not the first line, raise the error
                    raise ve
    return hadm_ids

# Function to load grouped predictions from a directory
def load_grouped_predictions(
    dataset_dir: Path,
    hadm_ids: Set[int],
    target_categories: List[str],
    merge_adjacent: bool,
    trim_annos: bool,
) -> Dict[int, Admission]:
    """Load dataset, do optional preprocessing/filtering and group evidence annotations by note_id"""
    dataset = MDACEData.from_dir(dataset_dir, require_text=True)

    dataset = filter_admissions_by_id(dataset, hadm_ids)

    if target_categories:
        dataset = filter_notes_by_category(dataset, set(target_categories))

    if merge_adjacent:
        dataset = merge_adjacent_annotations(dataset)

    if trim_annos:
        dataset = trim_annotations(dataset)

    return {adm.hadm_id: adm for adm in dataset.admissions}

# Main function that orchestrates the evaluation process
def main(args: argparse.Namespace):
    hadm_ids = load_hadm_ids(args.split_file)
    target_categories = args.note_category

    # Function to load grouped predictions
    def _load_grouped_predictions(data_dir: Path):
        return load_grouped_predictions(
            dataset_dir=data_dir,
            hadm_ids=hadm_ids,
            target_categories=target_categories,
            merge_adjacent=args.merge_adjacent,
            trim_annos=args.trim_annotations,
        )

    # Load gold standard annotations and predictions
    gold = _load_grouped_predictions(args.gold_dir)
    predictions = _load_grouped_predictions(args.predictions_dir)

    # Calculate error rates
    error_rates = AllErrorRates(tokenize_fn=tokenize)

    # Iterate through gold standard annotations and compare with predictions
    for hadm_id, actual_evidence in gold.items():
        predicted_evidence = predictions.get(hadm_id, [])
        if not predicted_evidence and actual_evidence:
            logger.debug(
                f"No evidence predicted for admission={hadm_id} [{len(actual_evidence):,} actual]"
            )

        error_rates.observe(actual_evidence, predicted_evidence)

    # Log error rates
    logger.info(error_rates)

    # Write results to a markdown file if specified
    md_out = args.md_out  # type: Path
    if md_out:
        md_out.parent.mkdir(parents=True, exist_ok=True)
        with open(md_out, "w", encoding="utf8") as ofp:
            print(str(error_rates), file=ofp)

# Entry point for script execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Define command-line arguments
    parser.add_argument(
        "--gold-dir",
        help="Path to directory containing annotation JSON files",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--predictions-dir",
        help="Path to directory containing annotation JSON files",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--split-file",
        help="Path to split CSV file",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--note-category",
        help="Only consider notes with this category; default considers all",
        type=str,
        nargs="*",
        action="extend",
        choices=[
            "Case Management",
            "Consult",
            "Discharge summary",
            "ECG",
            "General",
            "Nursing",
            "Nutrition",
            "Physician",
            "Radiology",
            "Rehab Services",
            "Respiratory",
            "RespiratoryGeneral",
        ],
    )

    parser.add_argument(
        "--md-out",
        help="Write results markdown to this file",
        type=Path,
        required=False,
    )

    cleanup = parser.add_argument_group("Clean Up Options")
    cleanup.add_argument(
        "--merge-adjacent",
        action="store_true",
        help="Merge annotations that are separated only by non-breaking characters",
    )
    cleanup.add_argument(
        "--trim-annotations",
        action="store_true",
        help="Trim non-breaking characters from annotations (e.g. punct/whitespace)",
    )

    # Configure logging level
    logging.basicConfig(level=logging.INFO)
    # Parse command-line arguments and execute the main function
    main(parser.parse_args())

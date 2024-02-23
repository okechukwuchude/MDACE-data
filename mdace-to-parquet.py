import argparse
import dataclasses
import logging
from pathlib import Path
from typing import Iterator, Dict

import pandas as pd

from mdace.data import MDACEData, Admission, Note, Annotation

# Setting up logging
logger = logging.getLogger(Path(__file__).name)


def find_annotations_dirs(data_dir: Path) -> Iterator[Path]:
    """
    Find annotation directories within the given data directory.
    """
    for specialty_dir in filter(lambda _: _.is_dir(), data_dir.iterdir()):
        for system_dir in filter(lambda _: _.is_dir(), specialty_dir.iterdir()):
            version_dirs = sorted(filter(lambda _: _.is_dir(), system_dir.iterdir()))
            annotations_dir = version_dirs[-1]
            if len(version_dirs) > 1:
                logger.warning(
                    f"got more than one version. selecting {annotations_dir.name}"
                )
            yield annotations_dir


def _get_out_path(out_dir: Path, annotations_dir: Path, extension: str) -> Path:
    """
    Generate output path for the serialized dataset.
    """
    code_system = annotations_dir.parent.name
    specialty = annotations_dir.parent.parent.name

    return out_dir / f"{specialty}-{code_system}.{extension}"


def _flatten(admission: Admission, note: Note, annotation: Annotation) -> Dict:
    """
    Flatten admission, note, and annotation into a dictionary.
    """
    adm_dict = dataclasses.asdict(admission)
    del adm_dict["notes"]
    note_dict = dataclasses.asdict(note)
    del note_dict["annotations"]
    anno_dict = dataclasses.asdict(annotation)
    span_dict = anno_dict.pop("span")
    billing_code_dict = anno_dict.pop("billing_code")

    return {
        **adm_dict,
        **note_dict,
        **span_dict,
        **billing_code_dict,
        **anno_dict,
    }


def as_dataframe(dataset: MDACEData) -> pd.DataFrame:
    """
    Convert dataset into a pandas DataFrame.
    """
    return pd.DataFrame(data=(_flatten(*tpl) for tpl in dataset))


def convert_and_serialize(target_dir: Path, output_dir: Path):
    """
    Convert MDACE data in the given directory into a DataFrame and serialize it as Parquet.
    """
    dataset = MDACEData.from_dir(target_dir)
    df = as_dataframe(dataset)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = _get_out_path(output_dir, target_dir, "parquet")
    df.to_parquet(output_file)


def main(args: argparse.Namespace):
    """
    Main function to convert MDACE data into Parquet format.
    """
    for target_dir in find_annotations_dirs(args.data_dir):
        convert_and_serialize(target_dir, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Define command-line arguments
    parser.add_argument(
        "--data-dir",
        help="Path to directory containing annotation JSON files",
        type=Path,
        default="with_text/gold",
    )
    parser.add_argument(
        "--output-dir", help="Path to write CSV file", type=Path, default="parquet"
    )

    # Configure logging level
    logging.basicConfig(level=logging.INFO)
    # Parse command-line arguments and execute the main function
    main(parser.parse_args())

import dataclasses
import re
from typing import List, Callable

from mdace.data import Span, Annotation, Admission

# Regular expression pattern to match word tokens
TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE | re.MULTILINE | re.DOTALL)


def tokenize(text: str) -> List[Span]:
    """Tokenizes the given text into word tokens."""
    
    # Find all matches of the token pattern in the lowercase text
    matches = TOKEN_PATTERN.finditer(text.lower())

    # Create Span objects for each match
    spans = [Span(*match.span(), covered_text=match.group()) for match in matches]

    # Exclude numbers greater than 10 as per Mullenbach's recommendation
    spans = [
        span
        for span in spans
        if not span.covered_text.isdigit() or int(span.covered_text) <= 10
    ]

    return spans


def tokenize_annotation(
    annotation: Annotation, tokenize_fn: Callable[[str], List[Span]]
) -> List[Annotation]:
    """Tokenizes the covered text of an annotation."""
    
    # Check if the covered text is available for tokenization
    if annotation.span.covered_text is None:
        raise ValueError(
            "Cannot tokenize annotations without text -- run inject-note-text.py"
        )

    token_offset = annotation.span.begin
    
    # Tokenize the covered text and adjust span positions
    return [
        dataclasses.replace(
            annotation,
            span=dataclasses.replace(
                span, begin=token_offset + span.begin, end=token_offset + span.end
            ),
        )
        for span in tokenize_fn(annotation.span.covered_text)
    ]


def tokenize_annotations(
    annotations: List[Annotation], tokenize_fn: Callable[[str], List[Span]]
) -> List[Annotation]:
    """Tokenizes a list of annotations."""
    
    flat = list()
    for a in annotations:
        flat.extend(tokenize_annotation(a, tokenize_fn))
    return flat


def tokenize_admission(
    admission: Admission, tokenize_fn: Callable[[str], List[Span]]
) -> Admission:
    """Tokenizes all annotations within an admission."""
    
    return dataclasses.replace(
        admission,
        notes=[
            dataclasses.replace(
                note, annotations=tokenize_annotations(note.annotations, tokenize_fn)
            )
            for note in admission.notes
        ],
    )

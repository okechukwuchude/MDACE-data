import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Callable, Hashable

from mdace.data import Annotation, Admission, Note, Span
from mdace.text import tokenize_admission

# Setting up logger
_logger = logging.getLogger(Path(__file__).name)

# Dataclass for representing error rates
@dataclass()
class ErrorRate:
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    # Overloading addition operator
    def __add__(self, other):
        if isinstance(other, ErrorRate):
            return ErrorRate(
                self.true_positives + other.true_positives,
                self.false_positives + other.false_positives,
                self.false_negatives + other.false_negatives,
            )
        else:
            raise ValueError(f"Cannot add {type(self)} and {type(other)}")

    # Property to calculate total count
    @property
    def total(self):
        return self.true_positives + self.false_positives + self.false_negatives

    # Property to calculate number of predicted values
    @property
    def n_pred(self):
        return self.true_positives + self.false_positives

    # Property to calculate number of actual values
    @property
    def n_actual(self):
        return self.true_positives + self.false_negatives

    # Property to calculate precision
    @property
    def precision(self) -> float:
        return self.true_positives / (self.true_positives + self.false_positives + 1e-6)

    # Property to calculate recall
    @property
    def recall(self) -> float:
        return self.true_positives / (self.true_positives + self.false_negatives + 1e-6)

    # Property to calculate F1 score
    @property
    def f1_score(self) -> float:
        return (
            2 * (self.precision * self.recall) / (self.precision + self.recall + 1e-6)
        )

    # String representation of ErrorRate object
    def __str__(self):
        return "\n".join(
            (
                "|  #Prd  |  #Act  |   TP   |   FP   |   FN   |   Pr   |   Rc   |   F1   |",
                "| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |",
                f"|{self.n_pred:^8d}|{self.n_actual:^8d}|{self.true_positives:^8d}|{self.false_positives:^8d}|{self.false_negatives:^8d}|{self.precision:^8.1%}|{self.recall:^8.1%}|{self.f1_score:^8.1%}|",
            )
        )

# Function to count unique errors between actual and predicted annotations
def _count_unique_errors(
    actual: Admission,
    predicted: Admission,
    key: Callable[[Note, Annotation], Hashable],
) -> ErrorRate:
    actual_set = set(key(note, annotation) for note, annotation in actual)
    predicted_set = set(key(note, annotation) for note, annotation in predicted)

    tp = actual_set & predicted_set
    fp = predicted_set - actual_set
    fn = actual_set - predicted_set
    return ErrorRate(
        true_positives=len(tp),
        false_positives=len(fp),
        false_negatives=len(fn),
    )

# Function to calculate exact match error
def exact_match_error(actual: Admission, predicted: Admission) -> ErrorRate:
    def key_fn(note: Note, anno: Annotation) -> Hashable:
        return note.note_id, anno

    return _count_unique_errors(actual, predicted, key_fn)

# Function to normalize text
def normalize(text: str) -> str:
    return text.lower()

# Function to calculate position-independent error
def position_independent_error(actual: Admission, predicted: Admission) -> ErrorRate:
    def key_fn(note: Note, anno: Annotation) -> Hashable:
        return normalize(anno.span.covered_text), anno.billing_code

    return _count_unique_errors(actual, predicted, key_fn)

# Function to calculate exact match error on token level
def token_exact_match_error(
    actual: Admission,
    predicted: Admission,
    tokenize: Callable[[str], List[Span]],
) -> ErrorRate:
    a_tokenized = tokenize_admission(actual, tokenize)
    p_tokenized = tokenize_admission(predicted, tokenize)

    return exact_match_error(a_tokenized, p_tokenized)

# Function to calculate position-independent error on token level
def token_position_independent_error(
    actual: Admission,
    predicted: Admission,
    tokenize: Callable[[str], List[Span]],
) -> ErrorRate:
    a_tokenized = tokenize_admission(actual, tokenize)
    p_tokenized = tokenize_admission(predicted, tokenize)

    return position_independent_error(a_tokenized, p_tokenized)

# Class to hold multiple error rates
class AllErrorRates(object):
    """Wrapper object for a bunch of error rates"""

    def __init__(self, tokenize_fn: Callable[[str], List[Span]]):
        self.tokenize_fn = tokenize_fn
        self.error_rates = dict(
            span_exact_match=ErrorRate(),
            span_position_independent=ErrorRate(),
            token_exact_match=ErrorRate(),
            token_position_independent=ErrorRate(),
        )

    # Method to observe actual and predicted values and update error rates
    def observe(self, actual: Admission, predicted: Admission):
        self.error_rates["span_exact_match"] += exact_match_error(actual, predicted)
        self.error_rates["span_position_independent"] += position_independent_error(
            actual, predicted
        )
        self.error_rates["token_exact_match"] += token_exact_match_error(
            actual, predicted, self.tokenize_fn
        )
        self.error_rates[
            "token_position_independent"
        ] += token_position_independent_error(actual, predicted, self.tokenize_fn)

    # String representation of AllErrorRates object
    def __str__(self):
        return "\n".join(
            (
                "",
                "Exact Span Match",
                "================",
                str(self.error_rates["span_exact_match"]),
                "",
                "Position-Independent Span Match",
                "===============================",
                str(self.error_rates["span_position_independent"]),
                "",
                "Exact Token Match",
                "=================",
                str(self.error_rates["token_exact_match"]),
                "",
                "Position-Independent Token Match",
                "================================",
                str(self.error_rates["token_position_independent"]),
            )
        )

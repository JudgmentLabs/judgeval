from typing import List, Optional

from judgeval.data import Example, GroundTruthExample


def examples_to_ground_truths(examples: List[Example]) -> List[GroundTruthExample]:
    """
    Convert a list of `Example` objects to a list of `GroundTruthExample` objects.

    Args:
        examples (List[Example]): A list of `Example` objects to convert.

    Returns:
        List[GroundTruthExample]: A list of `GroundTruthExample` objects.
    """

    if not isinstance(examples, list):
        raise TypeError("Input should be a list of `Example` objects")

    ground_truths = []
    ground_truths = []
    for e in examples:
        g_truth = {
            "input": e.input,
            "actual_output": e.actual_output,
            "expected_output": e.expected_output,
            "context": e.context,
            "retrieval_context": e.retrieval_context,
            "tools_called": e.tools_called,
            "expected_tools": e.expected_tools,
        }
        ground_truths.append(GroundTruthExample(**g_truth))
    return ground_truths


def ground_truths_to_examples(
    ground_truths: List[GroundTruthExample], 
    _alias: Optional[str] = None,
    _id: Optional[str] = None,
    ) -> List[Example]:
    """
    Converts a list of `GroundTruthExample` objects to a list of `Example` objects.

    Args:
        ground_truths (List[GroundTruthExample]): A list of `GroundTruthExample` objects to convert.
        _alias (Optional[str]): The alias of the dataset.
        _id (Optional[str]): The ID of the dataset.

    Returns:
        List[Example]: A list of `Example` objects.
    """

    if not isinstance(ground_truths, list):
        raise TypeError("Input should be a list of `GroundTruthExample` objects")

    examples = []
    for index, ground_truth in enumerate(ground_truths):
        e = Example(
            input=ground_truth.input,
            actual_output=ground_truth.actual_output,
            expected_output=ground_truth.expected_output,
            context=ground_truth.context,
            retrieval_context=ground_truth.retrieval_context,
            additional_metadata=ground_truth.additional_metadata,
            tools_called=ground_truth.tools_called,
            expected_tools=ground_truth.expected_tools,
            comments=ground_truth.comments,
            _dataset_alias=_alias,
            _dataset_id=_id,
            _dataset_rank=index,
        )
        examples.append(e)
    return examples

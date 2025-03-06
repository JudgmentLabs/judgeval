import pytest

from judgeval.data import Example, GroundTruthExample
from judgeval.data.datasets.utils import examples_to_ground_truths, ground_truths_to_examples


@pytest.fixture
def sample_example() -> Example:
    return Example(
        input="test input",
        actual_output="actual result",
        expected_output="expected result",
        context=["some context"],
        retrieval_context=["retrieval info"],
        tools_called=["tool1", "tool2"],
        expected_tools=["tool1"],
        additional_metadata={"key": "value"},
    )

@pytest.fixture
def sample_ground_truth() -> GroundTruthExample:
    return GroundTruthExample(
        input="test input",
        actual_output="actual result",
        expected_output="expected result",
        context=["some context"],
        retrieval_context=["retrieval info"],
        tools_called=["tool1", "tool2"],
        expected_tools=["tool1"],
        additional_metadata={"key": "value"},
        comments="test comment"
    )


class TestExamplesToGroundTruths:
    def test_empty_list(self):
        """Test conversion of empty list."""
        result = examples_to_ground_truths([])
        assert isinstance(result, list)
        assert len(result) == 0

    def test_single_example(self, sample_example):
        """Test conversion of a single example."""
        result = examples_to_ground_truths([sample_example])
        assert len(result) == 1
        assert isinstance(result[0], GroundTruthExample)
        assert result[0].input == sample_example.input
        assert result[0].actual_output == sample_example.actual_output
        assert result[0].expected_output == sample_example.expected_output

    def test_multiple_examples(self, sample_example):
        """Test conversion of multiple examples."""
        examples = [sample_example, sample_example]
        result = examples_to_ground_truths(examples)
        assert len(result) == 2
        assert all(isinstance(gt, GroundTruthExample) for gt in result)

    def test_none_input(self):
        """Test handling of None input."""
        with pytest.raises(TypeError):
            examples_to_ground_truths(None)

    def test_invalid_input_type(self):
        """Test handling of invalid input type."""
        with pytest.raises(TypeError):
            examples_to_ground_truths("not a list")


class TestGroundTruthsToExamples:
    def test_empty_list(self):
        """Test conversion of empty list."""
        result = ground_truths_to_examples([])
        assert isinstance(result, list)
        assert len(result) == 0

    def test_single_ground_truth(self, sample_ground_truth):
        """Test conversion of a single ground truth."""
        result = ground_truths_to_examples([sample_ground_truth])
        assert len(result) == 1
        assert isinstance(result[0], Example)
        assert result[0].input == sample_ground_truth.input
        assert result[0].actual_output == sample_ground_truth.actual_output
        assert result[0].expected_output == sample_ground_truth.expected_output

    def test_multiple_ground_truths(self, sample_ground_truth):
        """Test conversion of multiple ground truths."""
        ground_truths = [sample_ground_truth, sample_ground_truth]
        result = ground_truths_to_examples(ground_truths)
        assert len(result) == 2
        assert all(isinstance(ex, Example) for ex in result)

    def test_none_input(self):
        """Test handling of None input."""
        with pytest.raises(TypeError):
            ground_truths_to_examples(None)

    def test_invalid_input_type(self):
        """Test handling of invalid input type."""
        with pytest.raises(TypeError):
            ground_truths_to_examples("not a list")

    def test_preserves_metadata(self, sample_ground_truth):
        """Test that all metadata is preserved during conversion."""
        result = ground_truths_to_examples([sample_ground_truth])[0]
        assert result.additional_metadata == sample_ground_truth.additional_metadata
        assert result.tools_called == sample_ground_truth.tools_called
        assert result.expected_tools == sample_ground_truth.expected_tools
        
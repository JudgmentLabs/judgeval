import pytest
import json
import pandas as pd
import yaml
from unittest.mock import Mock, patch, mock_open
from judgeval.data.datasets import EvalDataset, EvalDatasetClient
from judgeval.data import Example

@pytest.fixture
def sample_example():
    return Example(
        input="test input",
        actual_output="test output",
        expected_output="expected output",
        context=["context1", "context2"],
        retrieval_context=["retrieval1"],
        additional_metadata={"key": "value"},
        tools_called=["tool1"],
        expected_tools=["tool1", "tool2"],
        name="test example"
    )


@pytest.fixture
def dataset():
    return EvalDataset(judgment_api_key="test_key", organization_id="test_org")

@pytest.fixture
def eval_dataset_client():
    return EvalDatasetClient(judgment_api_key="test_key", organization_id="test_org")

def test_init():
    dataset = EvalDataset(judgment_api_key="test_key")
    assert dataset.judgment_api_key == "test_key"
    assert dataset.examples == []
    assert dataset._alias is None
    assert dataset._id is None

def test_add_example(dataset, sample_example):
    dataset.add_example(sample_example)
    assert len(dataset.examples) == 1
    assert dataset.examples[0] == sample_example

@patch('requests.post')
def test_push_success(mock_post, dataset, sample_example, eval_dataset_client):
    # Setup mock response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"_alias": "test_alias", "_id": "test_id"}
    mock_post.return_value = mock_response

    # Add example and push
    dataset.add_example(sample_example)
    result = eval_dataset_client.push(dataset, "test_alias")

    assert result is True
    assert dataset._alias == "test_alias"
    assert dataset._id == "test_id"
    mock_post.assert_called_once()

@patch('requests.post')
def test_push_server_error(mock_post, dataset, eval_dataset_client):
    mock_response = Mock()
    mock_response.status_code = 500
    mock_post.return_value = mock_response

    result = eval_dataset_client.push(dataset, "test_alias")
    assert result is False

    mock_post.assert_called_once()

@patch('requests.post')
def test_pull_success(mock_post, eval_dataset_client):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "examples": [{"input": "test", "actual_output": "test"}],
        "_alias": "test_alias",
        "_id": "test_id"
    }
    mock_post.return_value = mock_response

    pulled_dataset = eval_dataset_client.pull("test_alias")
    assert len(pulled_dataset.examples) == 1
    assert pulled_dataset._alias == "test_alias"
    assert pulled_dataset._id == "test_id"

@patch('builtins.open', new_callable=mock_open)
def test_add_from_json(mock_file, dataset):
    json_data = {
        "examples": [{"input": "test", "actual_output": "test"}],
    }
    mock_file.return_value.__enter__.return_value.read.return_value = json.dumps(json_data)

    dataset.add_from_json("test.json")
    assert len(dataset.examples) == 1

@patch('pandas.read_csv')
def test_add_from_csv(mock_read_csv, dataset):
    mock_df = pd.DataFrame({
        'input': ['test1', 'test2'],
        'actual_output': ['output1', 'output2'],
        'expected_output': ['expected1', 'expected2'],
        'context': ['ctx1', 'ctx2'],
        'retrieval_context': ['ret1', 'ret2'],
        'additional_metadata': ['{}', '{}'],
        'tools_called': ['tool1', 'tool2'],
        'expected_tools': ['tool1', 'tool2'],
        'name': ['name1', None],
        'comments': [None, 'comment2'],
        'source_file': [None, 'file2'],
        'example': [True, False],
        'trace_id': [None, '123'],
        'example_id': ['12345', '12345']
    })
    mock_read_csv.return_value = mock_df

    dataset.add_from_csv("test.csv")
    assert len(dataset.examples) == 1

def test_save_as_json(dataset, sample_example, tmp_path):
    dataset.add_example(sample_example)
    save_path = tmp_path / "test_dir"
    dataset.save_as("json", str(save_path), "test_save")
    
    assert (save_path / "test_save.json").exists()
    with open(save_path / "test_save.json") as f:
        saved_data = json.load(f)
        assert "examples" in saved_data

def test_save_as_csv(dataset, sample_example, tmp_path):
    dataset.add_example(sample_example)
    save_path = tmp_path / "test_dir"
    dataset.save_as("csv", str(save_path), "test_save")
    
    assert (save_path / "test_save.csv").exists()
    df = pd.read_csv(save_path / "test_save.csv")
    assert len(df) == 1
    assert "input" in df.columns

def test_save_as_yaml(dataset, sample_example, tmp_path):
    dataset.add_example(sample_example)
    save_path = tmp_path / "test_dir"
    dataset.save_as("yaml", str(save_path), "test_save")
    
    # Check if the YAML file exists
    yaml_file_path = save_path / "test_save.yaml"
    assert yaml_file_path.exists(), "YAML file was not created."

    # Load the YAML file and check its contents
    with open(yaml_file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)

    # Validate the structure of the YAML data
    assert "examples" in yaml_data, "YAML data does not contain 'examples' key."
    assert len(yaml_data["examples"]) == 1, "YAML data should contain one example."
    

def test_save_as_invalid_type(dataset):
    with pytest.raises(TypeError):
        dataset.save_as("invalid", "test_dir")

def test_iter_and_len(dataset, sample_example):
    dataset.add_example(sample_example)
    assert len(dataset) == 1
    examples = list(dataset)
    assert len(examples) == 1
    assert examples[0] == sample_example

def test_str_representation(dataset, sample_example):
    dataset.add_example(sample_example)
    str_rep = str(dataset)
    assert "EvalDataset" in str_rep
    assert "examples" in str_rep

# new UTs for dataset UX testing

def test_load_from_json():
    ex1 = Example(
        input="test input",
        actual_output="test output",
        expected_output="expected output",
        context=["context1", "context2"],
        retrieval_context=["retrieval1"],
        additional_metadata={"key": "value"},
        tools_called=["tool1"],
        expected_tools=["tool1", "tool2"],
        name="test example",
        trace_id="123"
    )

    dataset = EvalDataset()

    dataset.add_from_json("tests/data/datasets/sample_data/dataset.json")

    # We can't do the same comparison as above because the timestamps are different
    assert len(dataset.examples) == 1
    loaded_example = dataset.examples[0]
    assert loaded_example.input == ex1.input
    assert loaded_example.actual_output == ex1.actual_output
    assert loaded_example.expected_output == ex1.expected_output
    assert loaded_example.context == ex1.context
    assert loaded_example.retrieval_context == ex1.retrieval_context
    assert loaded_example.additional_metadata == ex1.additional_metadata
    assert loaded_example.tools_called == ex1.tools_called
    assert loaded_example.expected_tools == ex1.expected_tools
    assert loaded_example.name == ex1.name
    assert loaded_example.trace_id == ex1.trace_id


def test_load_from_csv():
    ex1 = Example(
        input="test input",
        actual_output="test output",
        expected_output="expected output",
        context=["context1", "context2"],
        retrieval_context=["retrieval1"],
        additional_metadata={"key": "value"},
        tools_called=["tool1"],
        expected_tools=["tool1", "tool2"],
        name="test example",
        trace_id="123",
        example_id="12345"
    )

    dataset = EvalDataset()

    dataset.add_from_csv("tests/data/datasets/sample_data/dataset.csv")
    assert dataset.examples == [ex1]

def test_load_from_yaml():
    ex1 = Example(
        input="test input",
        actual_output="test output",
        expected_output="expected output",
        context=["context1", "context2"],
        retrieval_context=["retrieval1"],
        additional_metadata={"key": "value"},
        tools_called=["tool1"],
        expected_tools=["tool1", "tool2"],
        name="test example",
        trace_id="123",
        example_id="12345"
    )

    dataset = EvalDataset()

    dataset.add_from_yaml("tests/data/datasets/sample_data/dataset.yaml")
    assert dataset.examples == [ex1]

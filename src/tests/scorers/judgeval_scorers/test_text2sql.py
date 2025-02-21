import pytest
from judgeval.data import Example
from judgeval.scorers.judgeval_scorers.classifiers.text2sql.text2sql_scorer import Text2SQLScorer

@pytest.fixture
def valid_example():
    return Example(
        input="Show me all employees who earn more than 50000",
        actual_output="SELECT * FROM employees WHERE salary > 50000",
        context=["""
        Table: employees
        Columns:
        - id (INT)
        - name (VARCHAR)
        - salary (INT)
        - department_id (INT)
        """
    ]
)

@pytest.fixture
def invalid_example():
    return Example(
        input="Show me all employees who earn more than 50000",
        actual_output="SELECT * FROM employees WHERE salry > 50000",  # Typo in column name
        context=["""
        Table: employees
        Columns:
        - id (INT)
        - name (VARCHAR)
        - salary (INT)
        - department_id (INT)
        """]
    )

@pytest.fixture
def mismatched_example():
    return Example(
        input="Show me all employees who earn more than 50000",
        actual_output="SELECT * FROM employees WHERE salary < 50000",  # Wrong comparison operator
        context=["""
        Table: employees
        Columns:
        - id (INT)
        - name (VARCHAR)
        - salary (INT)
        - department_id (INT)
        """]
    )

def test_text2sql_scorer_initialization():
    """Test that the Text2SQL scorer initializes with correct default values"""
    assert Text2SQLScorer.name == "Text to SQL"
    assert Text2SQLScorer.threshold == 1.0
    assert Text2SQLScorer.options == {"Y": 1.0, "N": 0.0}
    assert len(Text2SQLScorer.conversation) == 1
    assert Text2SQLScorer.conversation[0]["role"] == "system"


def test_text2sql_scorer_update_threshold():
    """Test that the threshold can be updated"""
    original_threshold = Text2SQLScorer.threshold
    Text2SQLScorer.update_threshold(0.8)
    assert Text2SQLScorer.threshold == 0.8
    Text2SQLScorer.update_threshold(original_threshold)  # Reset to original

def test_text2sql_scorer_update_conversation():
    """Test that the conversation prompt can be updated"""
    new_conversation = [{
        "role": "system",
        "content": "New test conversation"
    }]
    Text2SQLScorer.update_conversation(new_conversation)
    assert Text2SQLScorer.conversation == new_conversation

def test_text2sql_scorer_update_options():
    """Test that the scoring options can be updated"""
    new_options = {"YES": 1.0, "NO": 0.0}
    Text2SQLScorer.update_options(new_options)
    assert Text2SQLScorer.options == new_options

def test_text2sql_scorer_serialization():
    """Test that the scorer can be properly serialized"""
    serialized = Text2SQLScorer.model_dump()
    assert "name" in serialized
    assert "score_type" in serialized
    assert "conversation" in serialized
    assert "options" in serialized
    assert "threshold" in serialized

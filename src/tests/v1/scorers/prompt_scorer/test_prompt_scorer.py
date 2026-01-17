from judgeval.v1.scorers.prompt_scorer.prompt_scorer import PromptScorer


def test_prompt_scorer_initialization():
    scorer = PromptScorer(name="TestPrompt", prompt="Test prompt text")
    assert scorer._name == "TestPrompt"
    assert scorer._prompt == "Test prompt text"
    assert scorer._threshold == 0.5
    assert scorer._options is None
    assert scorer._model is None
    assert scorer._description is None
    assert scorer._is_trace is False


def test_prompt_scorer_with_options():
    options = {"yes": 1.0, "no": 0.0}
    scorer = PromptScorer(name="TestPrompt", prompt="Test", options=options)
    assert scorer._options == {"yes": 1.0, "no": 0.0}


def test_prompt_scorer_with_threshold():
    scorer = PromptScorer(name="TestPrompt", prompt="Test", threshold=0.8)
    assert scorer._threshold == 0.8


def test_prompt_scorer_with_model():
    scorer = PromptScorer(name="TestPrompt", prompt="Test", model="gpt-4")
    assert scorer._model == "gpt-4"


def test_prompt_scorer_with_description():
    scorer = PromptScorer(
        name="TestPrompt", prompt="Test", description="Test description"
    )
    assert scorer._description == "Test description"


def test_prompt_scorer_trace_mode():
    scorer = PromptScorer(name="TestPrompt", prompt="Test", is_trace=True)
    assert scorer._is_trace is True


def test_prompt_scorer_get_name():
    scorer = PromptScorer(name="MyPrompt", prompt="Test")
    assert scorer.get_name() == "MyPrompt"


def test_prompt_scorer_get_prompt():
    scorer = PromptScorer(name="Test", prompt="Sample prompt")
    assert scorer.get_prompt() == "Sample prompt"


def test_prompt_scorer_get_threshold():
    scorer = PromptScorer(name="Test", prompt="Test", threshold=0.7)
    assert scorer.get_threshold() == 0.7


def test_prompt_scorer_get_options():
    options = {"a": 1.0, "b": 0.5}
    scorer = PromptScorer(name="Test", prompt="Test", options=options)
    retrieved_options = scorer.get_options()
    assert retrieved_options == {"a": 1.0, "b": 0.5}


def test_prompt_scorer_get_options_returns_copy():
    options = {"a": 1.0}
    scorer = PromptScorer(name="Test", prompt="Test", options=options)
    retrieved = scorer.get_options()
    retrieved["a"] = 0.5
    assert scorer._options["a"] == 1.0


def test_prompt_scorer_get_model():
    scorer = PromptScorer(name="Test", prompt="Test", model="claude-3")
    assert scorer.get_model() == "claude-3"


def test_prompt_scorer_get_description():
    scorer = PromptScorer(name="Test", prompt="Test", description="My description")
    assert scorer.get_description() == "My description"


def test_prompt_scorer_set_threshold():
    scorer = PromptScorer(name="Test", prompt="Test")
    scorer.set_threshold(0.9)
    assert scorer._threshold == 0.9


def test_prompt_scorer_set_prompt():
    scorer = PromptScorer(name="Test", prompt="Initial")
    scorer.set_prompt("Updated prompt")
    assert scorer._prompt == "Updated prompt"


def test_prompt_scorer_set_model():
    scorer = PromptScorer(name="Test", prompt="Test")
    scorer.set_model("gpt-4o")
    assert scorer._model == "gpt-4o"


def test_prompt_scorer_set_options():
    scorer = PromptScorer(name="Test", prompt="Test")
    options = {"yes": 1.0, "no": 0.0}
    scorer.set_options(options)
    assert scorer._options == {"yes": 1.0, "no": 0.0}


def test_prompt_scorer_set_options_copies():
    scorer = PromptScorer(name="Test", prompt="Test")
    options = {"yes": 1.0}
    scorer.set_options(options)
    options["yes"] = 0.5
    assert scorer._options["yes"] == 1.0


def test_prompt_scorer_set_description():
    scorer = PromptScorer(name="Test", prompt="Test")
    scorer.set_description("New description")
    assert scorer._description == "New description"


def test_prompt_scorer_get_scorer_config():
    scorer = PromptScorer(
        name="TestScorer",
        prompt="Test prompt",
        threshold=0.6,
        options={"yes": 1.0, "no": 0.0},
        model="gpt-4",
        description="Test description",
    )

    config = scorer.get_scorer_config()

    assert config["score_type"] == "Prompt Scorer"
    assert config["threshold"] == 0.6
    assert config["name"] == "TestScorer"
    assert config["kwargs"]["prompt"] == "Test prompt"
    assert config["kwargs"]["options"] == {"yes": 1.0, "no": 0.0}
    assert config["kwargs"]["model"] == "gpt-4"
    assert config["kwargs"]["description"] == "Test description"


def test_prompt_scorer_get_scorer_config_trace_mode():
    scorer = PromptScorer(name="TraceScorer", prompt="Test", is_trace=True)

    config = scorer.get_scorer_config()
    assert config["score_type"] == "Trace Prompt Scorer"


def test_prompt_scorer_get_scorer_config_minimal():
    scorer = PromptScorer(name="Minimal", prompt="Test")

    config = scorer.get_scorer_config()

    assert config["score_type"] == "Prompt Scorer"
    assert config["kwargs"]["prompt"] == "Test"
    assert "options" not in config["kwargs"]
    assert "model" not in config["kwargs"]
    assert "description" not in config["kwargs"]

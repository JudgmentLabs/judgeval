from judgeval.v1.judges.prompt_judge import PromptJudge


def test_prompt_judge_initialization():
    judge = PromptJudge(name="TestPrompt", prompt="Test prompt text")
    assert judge._name == "TestPrompt"
    assert judge._prompt == "Test prompt text"
    assert judge._threshold == 0.5
    assert judge._options is None
    assert judge._model is None
    assert judge._description is None


def test_prompt_judge_with_options():
    options = {"yes": 1.0, "no": 0.0}
    judge = PromptJudge(name="TestPrompt", prompt="Test", options=options)
    assert judge._options == {"yes": 1.0, "no": 0.0}


def test_prompt_judge_with_threshold():
    judge = PromptJudge(name="TestPrompt", prompt="Test", threshold=0.8)
    assert judge._threshold == 0.8


def test_prompt_judge_with_model():
    judge = PromptJudge(name="TestPrompt", prompt="Test", model="gpt-4")
    assert judge._model == "gpt-4"


def test_prompt_judge_with_description():
    judge = PromptJudge(
        name="TestPrompt", prompt="Test", description="Test description"
    )
    assert judge._description == "Test description"


def test_prompt_judge_get_name():
    judge = PromptJudge(name="MyPrompt", prompt="Test")
    assert judge.get_name() == "MyPrompt"


def test_prompt_judge_get_prompt():
    judge = PromptJudge(name="Test", prompt="Sample prompt")
    assert judge.get_prompt() == "Sample prompt"


def test_prompt_judge_get_threshold():
    judge = PromptJudge(name="Test", prompt="Test", threshold=0.7)
    assert judge.get_threshold() == 0.7


def test_prompt_judge_get_options():
    options = {"a": 1.0, "b": 0.5}
    judge = PromptJudge(name="Test", prompt="Test", options=options)
    retrieved_options = judge.get_options()
    assert retrieved_options == {"a": 1.0, "b": 0.5}


def test_prompt_judge_get_options_returns_copy():
    options = {"a": 1.0}
    judge = PromptJudge(name="Test", prompt="Test", options=options)
    retrieved = judge.get_options()
    retrieved["a"] = 0.5
    assert judge._options["a"] == 1.0


def test_prompt_judge_get_model():
    judge = PromptJudge(name="Test", prompt="Test", model="claude-3")
    assert judge.get_model() == "claude-3"


def test_prompt_judge_get_description():
    judge = PromptJudge(name="Test", prompt="Test", description="My description")
    assert judge.get_description() == "My description"


def test_prompt_judge_set_threshold():
    judge = PromptJudge(name="Test", prompt="Test")
    judge.set_threshold(0.9)
    assert judge._threshold == 0.9


def test_prompt_judge_set_prompt():
    judge = PromptJudge(name="Test", prompt="Initial")
    judge.set_prompt("Updated prompt")
    assert judge._prompt == "Updated prompt"


def test_prompt_judge_set_model():
    judge = PromptJudge(name="Test", prompt="Test")
    judge.set_model("gpt-4o")
    assert judge._model == "gpt-4o"


def test_prompt_judge_set_options():
    judge = PromptJudge(name="Test", prompt="Test")
    options = {"yes": 1.0, "no": 0.0}
    judge.set_options(options)
    assert judge._options == {"yes": 1.0, "no": 0.0}


def test_prompt_judge_set_options_copies():
    judge = PromptJudge(name="Test", prompt="Test")
    options = {"yes": 1.0}
    judge.set_options(options)
    options["yes"] = 0.5
    assert judge._options["yes"] == 1.0


def test_prompt_judge_set_description():
    judge = PromptJudge(name="Test", prompt="Test")
    judge.set_description("New description")
    assert judge._description == "New description"


def test_prompt_judge_get_scorer_config():
    judge = PromptJudge(
        name="TestJudge",
        prompt="Test prompt",
        threshold=0.6,
        options={"yes": 1.0, "no": 0.0},
        model="gpt-4",
        description="Test description",
    )

    config = judge.get_scorer_config()

    assert config["score_type"] == "Prompt Scorer"
    assert config["threshold"] == 0.6
    assert config["name"] == "TestJudge"
    assert config["kwargs"]["prompt"] == "Test prompt"
    assert config["kwargs"]["options"] == {"yes": 1.0, "no": 0.0}
    assert config["kwargs"]["model"] == "gpt-4"
    assert config["kwargs"]["description"] == "Test description"


def test_prompt_judge_get_scorer_config_minimal():
    judge = PromptJudge(name="Minimal", prompt="Test")

    config = judge.get_scorer_config()

    assert config["score_type"] == "Prompt Scorer"
    assert config["kwargs"]["prompt"] == "Test"
    assert "options" not in config["kwargs"]
    assert "model" not in config["kwargs"]
    assert "description" not in config["kwargs"]

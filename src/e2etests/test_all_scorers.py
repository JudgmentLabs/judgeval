"""
base e2e tests for all default judgeval scorers
"""


from judgeval.judgment_client import JudgmentClient
from judgeval.scorers import (
    AnswerCorrectnessScorer,
    AnswerRelevancyScorer, 
    ContextualPrecisionScorer,
    ContextualRecallScorer,
    ContextualRelevancyScorer,
    FaithfulnessScorer,
    HallucinationScorer,
    SummarizationScorer,
    ComparisonScorer,
    Text2SQLScorer,
    InstructionAdherenceScorer,
)

from judgeval.data import Example


def test_ac_scorer():
    
    example = Example(
        input="What's the capital of France?",
        actual_output="The capital of France is Paris.",
        expected_output="France's capital is Paris. It used to be called the city of lights until 1968.",
    )

    scorer = AnswerCorrectnessScorer(threshold=0.5)

    client = JudgmentClient()
    PROJECT_NAME = "test-project"
    EVAL_RUN_NAME = "test-run-ac"
    
    # Test with use_judgment=True
    res = client.run_evaluation(
        examples=[example],
        scorers=[scorer],
        model="Qwen/Qwen2.5-72B-Instruct-Turbo",
        log_results=True,
        project_name=PROJECT_NAME,
        eval_run_name=EVAL_RUN_NAME,
        use_judgment=True,
        override=True,
    )
    print_debug_on_failure(res[0])

    # Test with use_judgment=False 
    res = client.run_evaluation(
        examples=[example],
        scorers=[scorer],
        model="Qwen/Qwen2.5-72B-Instruct-Turbo",
        log_results=True,
        project_name=PROJECT_NAME,
        eval_run_name=EVAL_RUN_NAME,
        use_judgment=False,
        override=True,
    )
    print_debug_on_failure(res[0])


def test_ar_scorer():

    example_1 = Example(  # should pass
        input="What's the capital of France?",
        actual_output="The capital of France is Paris."
    )

    example_2 = Example(  # should fail
        input="What's the capital of France?",
        actual_output="There's alot to do in Marseille. Lots of bars, restaurants, and museums."
    )

    scorer = AnswerRelevancyScorer(threshold=0.5)

    client = JudgmentClient()
    PROJECT_NAME = "test-project"
    EVAL_RUN_NAME = "test-run-ar"

    # Test with use_judgment=True
    res = client.run_evaluation(
        examples=[example_1, example_2],
        scorers=[scorer],
        model="Qwen/Qwen2.5-72B-Instruct-Turbo",
        log_results=True,
        project_name=PROJECT_NAME,
        eval_run_name=EVAL_RUN_NAME,
        use_judgment=True,
        override=True,
    )

    print_debug_on_failure(res[0])
    print_debug_on_failure(res[1])
    
    assert res[0].success == True
    assert res[1].success == False

    # Test with use_judgment=False
    res = client.run_evaluation(
        examples=[example_1, example_2],
        scorers=[scorer],
        model="Qwen/Qwen2.5-72B-Instruct-Turbo",
        log_results=True,
        project_name=PROJECT_NAME,
        eval_run_name=EVAL_RUN_NAME,
        use_judgment=False,
        override=True,
    )

    print_debug_on_failure(res[0])
    print_debug_on_failure(res[1])
    
    assert res[0].success == True
    assert res[1].success == False


def test_comparison_scorer():
    example_1 = Example(
        input="Generate a poem about a field",
        expected_output="A sunlit meadow, alive with whispers of wind, where daisies dance and hope begins again. Each petal holds a promise—bright, unbruised— a symphony of light that cannot be refused.",
        actual_output="A field, kinda windy, with some flowers, stuff growing, and maybe a nice vibe. Petals do things, I guess? Like, they're there… and light exists, but whatever, it's fine."
    )   

    example_2 = Example(
        input="Generate a poem about a field",
        expected_output="A field, kinda windy, with some flowers, stuff growing, and maybe a nice vibe. Petals do things, I guess? Like, they're there… and light exists, but whatever, it's fine.",
        actual_output="A field, kinda windy, with some flowers, stuff growing, and maybe a nice vibe. Petals do things, I guess? Like, they're there… and light exists, but whatever, it's fine."
    )

    scorer = ComparisonScorer(threshold=1, criteria="Tone and Style", description="The tone and style of the poem should be consistent and cohesive.")

    client = JudgmentClient()
    PROJECT_NAME = "test-project"
    EVAL_RUN_NAME = "test-run-comparison"

    res = client.run_evaluation(
        examples=[example_1, example_2],
        scorers=[scorer],
        model="Qwen/Qwen2.5-72B-Instruct-Turbo",
        log_results=True,
        project_name=PROJECT_NAME,
        eval_run_name=EVAL_RUN_NAME,
        use_judgment=True,
        override=True,  
    )

    print_debug_on_failure(res[1])
    assert res[0].success == False
    assert res[1].success == True

def test_cp_scorer():

    example_1 = Example(  # should pass
        input="What's the capital of France?",
        actual_output="The capital of France is Paris.",
        expected_output="France's capital is Paris. It used to be called the city of lights until 1968.",
        retrieval_context=[
            "Paris is a city in central France. It is the capital of France.",
            "Paris is well known for its museums, architecture, and cuisine.",
            "Flights to Paris are available from San Francisco starting at $1000."
        ]
    )

    example_2 = Example(
        input="What's the capital of France?",
        actual_output="There's alot to do in Marseille. Lots of bars, restaurants, and museums.",
        expected_output="France's capital is Paris. It used to be called the city of lights until 1968.",
        retrieval_context=[
            "Marseille is a city in southern France. It is the second largest city in France.",
            "Marseille is known for its beaches, historic port, and vibrant nightlife.",
            "Flights to Marseille are available from San Francisco starting at $500."
        ]
    )

    scorer = ContextualPrecisionScorer(threshold=0.5)

    client = JudgmentClient()
    PROJECT_NAME = "test-project"
    EVAL_RUN_NAME = "test-run-cp"

    # Test with use_judgment=True
    res = client.run_evaluation(
        examples=[example_1, example_2],
        scorers=[scorer],
        model="Qwen/Qwen2.5-72B-Instruct-Turbo",
        log_results=True,
        project_name=PROJECT_NAME,
        eval_run_name=EVAL_RUN_NAME,
        use_judgment=True,
        override=True,
    )

    print_debug_on_failure(res[0])
    print_debug_on_failure(res[1])

    assert res[0].success == True  # example_1 should pass
    assert res[1].success == False  # example_2 should fail

    # Test with use_judgment=False
    res = client.run_evaluation(
        examples=[example_1, example_2],
        scorers=[scorer],
        model="Qwen/Qwen2.5-72B-Instruct-Turbo",
        log_results=True,
        project_name=PROJECT_NAME,
        eval_run_name=EVAL_RUN_NAME,
        use_judgment=False,
        override=True,
    )

    print_debug_on_failure(res[0])
    print_debug_on_failure(res[1])

    assert res[0].success == True  # example_1 should pass
    assert res[1].success == False  # example_2 should fail


def test_cr_scorer():

    example_1 = Example(  # should pass
        input="What's the capital of France?",
        actual_output="The capital of France is Paris.",
        expected_output="France's capital is Paris. It used to be called the city of lights until 1968.",
        retrieval_context=[
            "Paris is a city in central France. It is the capital of France.",
            "Paris is well known for its museums, architecture, and cuisine.",
            "Flights to Paris are available from San Francisco starting at $1000."
        ]
    )

    scorer = ContextualRecallScorer(threshold=0.5)

    client = JudgmentClient()
    PROJECT_NAME = "test-project"
    EVAL_RUN_NAME = "test-run-cr"

    # Test with use_judgment=True
    res = client.run_evaluation(
        examples=[example_1],
        scorers=[scorer],
        model="Qwen/Qwen2.5-72B-Instruct-Turbo",
        log_results=True,
        project_name=PROJECT_NAME,
        eval_run_name=EVAL_RUN_NAME,
        use_judgment=True,
        override=True,
    )

    print_debug_on_failure(res[0])

    assert res[0].success == True  # example_1 should pass

    # Test with use_judgment=False
    res = client.run_evaluation(
        examples=[example_1],
        scorers=[scorer],
        model="Qwen/Qwen2.5-72B-Instruct-Turbo",
        log_results=True,
        project_name=PROJECT_NAME,
        eval_run_name=EVAL_RUN_NAME,
        use_judgment=False,
        override=True,
    )

    print_debug_on_failure(res[0])

    assert res[0].success == True  # example_1 should pass


def test_crelevancy_scorer():

    example_1 = Example(  # should pass
        input="What's the capital of France?",
        actual_output="The capital of France is Paris.",
        expected_output="France's capital is Paris. It used to be called the city of lights until 1968.",
        retrieval_context=[
            "Paris is a city in central France. It is the capital of France.",
            "Paris is well known for its museums, architecture, and cuisine.",
            "Flights to Paris are available from San Francisco starting at $1000."
        ]
    )

    scorer = ContextualRelevancyScorer(threshold=0.5)

    client = JudgmentClient()
    PROJECT_NAME = "test-project"
    EVAL_RUN_NAME = "test-run-crelevancy"

    # Test with use_judgment=True
    res = client.run_evaluation(
        examples=[example_1],
        scorers=[scorer],
        model="Qwen/Qwen2.5-72B-Instruct-Turbo",
        log_results=True,
        project_name=PROJECT_NAME,
        eval_run_name=EVAL_RUN_NAME,
        use_judgment=True,
        override=True,
    )

    print(res)

    print_debug_on_failure(res[0])

    assert res[0].success == True  # example_1 should pass

    # Test with use_judgment=False
    res = client.run_evaluation(
        examples=[example_1],
        scorers=[scorer],
        model="Qwen/Qwen2.5-72B-Instruct-Turbo",
        log_results=True,
        project_name=PROJECT_NAME,
        eval_run_name=EVAL_RUN_NAME,
        use_judgment=False,
        override=True,
    )

    print_debug_on_failure(res[0])

    assert res[0].success == True  # example_1 should pass


def test_faithfulness_scorer():

    faithful_example = Example(  # should pass
        input="What's the capital of France?",
        actual_output="The capital of France is Paris.",
        expected_output="France's capital is Paris. It used to be called the city of lights until 1968.",
        retrieval_context=[
            "Paris is a city in central France. It is the capital of France.",
            "Paris is well known for its museums, architecture, and cuisine.",
            "Flights to Paris are available from San Francisco starting at $1000."
        ]
    )

    contradictory_example = Example(  # should fail
        input="What's the capital of France?",
        actual_output="The capital of France is Lyon. It's located in southern France near the Mediterranean coast.",
        expected_output="France's capital is Paris. It used to be called the city of lights until 1968.",
        retrieval_context=[
            "Paris is a city in central France. It is the capital of France.",
            "Paris is well known for its museums, architecture, and cuisine.",
            "Flights to Paris are available from San Francisco starting at $1000."
        ]
    )

    scorer = FaithfulnessScorer(threshold=1.0)

    client = JudgmentClient()
    PROJECT_NAME = "test-project"
    EVAL_RUN_NAME = "test-run-faithfulness"

    # Test with use_judgment=True
    res = client.run_evaluation(
        examples=[faithful_example, contradictory_example],
        scorers=[scorer],
        model="Qwen/Qwen2.5-72B-Instruct-Turbo",
        log_results=True,
        project_name=PROJECT_NAME,
        eval_run_name=EVAL_RUN_NAME,
        use_judgment=True,
        override=True,
    )

    print_debug_on_failure(res[0])
    print_debug_on_failure(res[1])

    assert res[0].success == True  # faithful_example should pass
    assert res[1].success == False, res[1]  # contradictory_example should fail

    # Test with use_judgment=False
    res = client.run_evaluation(
        examples=[faithful_example, contradictory_example],
        scorers=[scorer],
        model="Qwen/Qwen2.5-72B-Instruct-Turbo",
        log_results=True,
        project_name=PROJECT_NAME,
        eval_run_name=EVAL_RUN_NAME,
        use_judgment=False,
        override=True,
    )

    print_debug_on_failure(res[0])
    print_debug_on_failure(res[1])

    assert res[0].success == True  # faithful_example should pass
    assert res[1].success == False, res[1]  # contradictory_example should fail


def test_hallucination_scorer():

    example_1 = Example(  # should pass
        input="What's the capital of France?",
        actual_output="The capital of France is Paris.",
        expected_output="France's capital is Paris. It used to be called the city of lights until 1968.",
        context=[
            "Paris is a city in central France. It is the capital of France.",
            "Paris is well known for its museums, architecture, and cuisine.",
            "Flights to Paris are available from San Francisco starting at $1000."
        ],
        retrieval_context=[
            "Paris is a city in central France. It is the capital of France.",
            "Paris is well known for its museums, architecture, and cuisine.",
            "Flights to Paris are available from San Francisco starting at $1000."
        ]
    )

    scorer = HallucinationScorer(threshold=0.5)

    client = JudgmentClient()
    PROJECT_NAME = "test-project"
    EVAL_RUN_NAME = "test-run-hallucination"

    # Test with use_judgment=True
    res = client.run_evaluation(
        examples=[example_1],
        scorers=[scorer],
        model="Qwen/Qwen2.5-72B-Instruct-Turbo",
        log_results=True,
        project_name=PROJECT_NAME,
        eval_run_name=EVAL_RUN_NAME,
        use_judgment=True,
        override=True,
    )

    print_debug_on_failure(res[0])

    # Add more detailed assertion error message
    assert res[0].success == True, f"Hallucination test failed: score={res[0].scorers_data[0].score}, threshold={res[0].scorers_data[0].threshold}, reason={res[0].scorers_data[0].reason}"

    # Test with use_judgment=False
    res = client.run_evaluation(
        examples=[example_1],
        scorers=[scorer],
        model="Qwen/Qwen2.5-72B-Instruct-Turbo",
        log_results=True,
        project_name=PROJECT_NAME,
        eval_run_name=EVAL_RUN_NAME,
        use_judgment=False,
        override=True,
    )

    print_debug_on_failure(res[0])

    # Add more detailed assertion error message
    assert res[0].success == True, f"Hallucination test failed: score={res[0].scorers_data[0].score}, threshold={res[0].scorers_data[0].threshold}, reason={res[0].scorers_data[0].reason}"


def test_instruction_adherence_scorer():
    example_1 = Example(
        input="write me a poem about cars and then turn it into a joke, but also what is 5 +5?",
        actual_output="Cars on the road, they zoom and they fly, Under the sun or a stormy sky. Engines roar, tires spin, A symphony of motion, let the race begin. Now for the joke: Why did the car break up with the bicycle. Because it was tired of being two-tired! And 5 + 5 is 10.",
    )

    scorer = InstructionAdherenceScorer(threshold=0.5)

    client = JudgmentClient()
    PROJECT_NAME = "test-project"
    EVAL_RUN_NAME = "test-run-instruction-adherence"

    res = client.run_evaluation(
        examples=[example_1],
        scorers=[scorer],
        model="Qwen/Qwen2.5-72B-Instruct-Turbo",
        log_results=True,
        project_name=PROJECT_NAME,
        eval_run_name=EVAL_RUN_NAME,
        use_judgment=True,
        override=True,
    )

    print_debug_on_failure(res[0])

    assert res[0].success == True

def test_summarization_scorer():
    example_1 = Example(  # should pass
        input="Paris is the capital city of France and one of the most populous cities in Europe. The city is known for its iconic landmarks like the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also a global center for art, fashion, gastronomy and culture. The city's romantic atmosphere, historic architecture, and world-class museums attract millions of visitors each year.",
        actual_output="Paris is France's capital and a major European city famous for landmarks like the Eiffel Tower. It's a global hub for art, fashion and culture that draws many tourists.",
    )

    scorer = SummarizationScorer(threshold=0.5)

    client = JudgmentClient()
    PROJECT_NAME = "test-project"
    EVAL_RUN_NAME = "test-run-summarization"

    # Test with use_judgment=True
    res = client.run_evaluation(
        examples=[example_1],
        scorers=[scorer],
        model="Qwen/Qwen2.5-72B-Instruct-Turbo",
        log_results=True,
        project_name=PROJECT_NAME,
        eval_run_name=EVAL_RUN_NAME,
        use_judgment=True,
        override=True,
    )

    print_debug_on_failure(res[0])
    
    # Add detailed assertion error message
    assert res[0].success == True, f"Summarization test failed: score={res[0].scorers_data[0].score}, threshold={res[0].scorers_data[0].threshold}, reason={res[0].scorers_data[0].reason}"

    # Test with use_judgment=False
    res = client.run_evaluation(
        examples=[example_1],
        scorers=[scorer],
        model="Qwen/Qwen2.5-72B-Instruct-Turbo",
        log_results=True,
        project_name=PROJECT_NAME,
        eval_run_name=EVAL_RUN_NAME,
        use_judgment=False,
        override=True,
    )

    print_debug_on_failure(res[0])
    
    # Add detailed assertion error message
    assert res[0].success == True, f"Summarization test failed: score={res[0].scorers_data[0].score}, threshold={res[0].scorers_data[0].threshold}, reason={res[0].scorers_data[0].reason}"


def test_text2sql_scorer():
    client = JudgmentClient()

    table_schema = """CREATE TABLE Artists (
    artist_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    genre VARCHAR(100),
    followers INT,
    popularity INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE Albums (
    album_id VARCHAR(50) PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    artist_id VARCHAR(50) NOT NULL,
    release_date DATE,
    total_tracks INT,
    album_type VARCHAR(50) CHECK (album_type IN ('album', 'single', 'compilation')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (artist_id) REFERENCES Artists(artist_id) ON DELETE CASCADE
);

CREATE TABLE Tracks (
    track_id VARCHAR(50) PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    album_id VARCHAR(50) NOT NULL,
    artist_id VARCHAR(50) NOT NULL,
    duration_ms INT NOT NULL,
    explicit BOOLEAN DEFAULT FALSE,
    popularity INT DEFAULT 0,
    preview_url VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (album_id) REFERENCES Albums(album_id) ON DELETE CASCADE,
    FOREIGN KEY (artist_id) REFERENCES Artists(artist_id) ON DELETE CASCADE
);

CREATE TABLE Users (
    user_id VARCHAR(50) PRIMARY KEY,
    username VARCHAR(100) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE Playlists (
    playlist_id VARCHAR(50) PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES Users(user_id) ON DELETE CASCADE
);

CREATE TABLE PlaylistTracks (
    playlist_id VARCHAR(50),
    track_id VARCHAR(50),
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (playlist_id, track_id),
    FOREIGN KEY (playlist_id) REFERENCES Playlists(playlist_id) ON DELETE CASCADE,
    FOREIGN KEY (track_id) REFERENCES Tracks(track_id) ON DELETE CASCADE
);

CREATE TABLE UserListeningHistory (
    history_id SERIAL PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    track_id VARCHAR(50) NOT NULL,
    listened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES Users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (track_id) REFERENCES Tracks(track_id) ON DELETE CASCADE
);
"""

    all_tracks_one_artist_correct = Example(
        input="Find all tracks by the artist 'Drake', sorted by popularity.",
        actual_output="""SELECT t.track_id, t.title, t.popularity, a.name AS artist_name
FROM Tracks t
JOIN Artists a ON t.artist_id = a.artist_id
WHERE a.name = 'Drake'
ORDER BY t.popularity DESC;
""",
        retrieval_context=[table_schema]
    )

    most_listened_to_one_user_correct = Example(
        input="Find the most listened to track by user 'user123'.",
        actual_output="""SELECT t.track_id, t.title, COUNT(uh.history_id) AS play_count
FROM UserListeningHistory uh
JOIN Tracks t ON uh.track_id = t.track_id
WHERE uh.user_id = 'user123'
GROUP BY t.track_id, t.title
ORDER BY play_count DESC
LIMIT 1;
""",
        retrieval_context=[table_schema]
    )

    highest_num_playlists_correct = Example(
        input="Find the 5 users with the highest number of playlists.",
        actual_output="""SELECT u.user_id, u.username, COUNT(p.playlist_id) AS total_playlists
FROM Users u
JOIN Playlists p ON u.user_id = p.user_id
GROUP BY u.user_id, u.username
ORDER BY total_playlists DESC
LIMIT 5;
""",
        retrieval_context=[table_schema]
    )

    most_popular_tracks_all_users_correct = Example(
        input="Find the 10 most popular tracks across all users.",
        actual_output="""SELECT t.track_id, t.title, COUNT(uh.history_id) AS total_listens
FROM Tracks t
JOIN UserListeningHistory uh ON t.track_id = uh.track_id
GROUP BY t.track_id, t.title
ORDER BY total_listens DESC
LIMIT 10;
""",
        retrieval_context=[table_schema]
    )

    most_popular_tracks_all_users_incorrect = Example(
        input="Find the 10 most popular tracks across all users.",
        actual_output="""SELECT t.track_user, t.title, COUNT(uh.history_id) AS total_listens
FROM Tracks t
JOIN UserHistory uh ON t.track_user = uh.track_user
GROUP BY t.track_user, t.title
ORDER BY total_listens DESC
LIMIT 10;
""",
        retrieval_context=[table_schema]
    )


    res = client.run_evaluation(
        examples=[
            all_tracks_one_artist_correct, 
            most_listened_to_one_user_correct, 
            highest_num_playlists_correct, 
            most_popular_tracks_all_users_correct, 
            most_popular_tracks_all_users_incorrect
        ],
        scorers=[Text2SQLScorer],
        model="gpt-4o-mini",
        project_name="text2sql",
        eval_run_name="text2sql_test",
        override=True
    )

    assert print_debug_on_failure(res[0])
    assert print_debug_on_failure(res[1])
    assert print_debug_on_failure(res[2])
    assert print_debug_on_failure(res[3])
    assert not print_debug_on_failure(res[4])


def print_debug_on_failure(result) -> bool:
    """
    Helper function to print debug info only on test failure
    
    Returns:
        bool: True if the test passed, False if it failed
    """
    if not result.success:
        print("\n=== Test Failure Details ===")
        print(f"Input: {result.input}")
        print(f"Output: {result.actual_output}")
        print(f"Success: {result.success}")
        if hasattr(result, 'retrieval_context'):
            print(f"Retrieval Context: {result.retrieval_context}")
        print("\nScorer Details:")
        for scorer_data in result.scorers_data:
            print(f"- Name: {scorer_data.name}")
            print(f"- Score: {scorer_data.score}")
            print(f"- Threshold: {scorer_data.threshold}")
            print(f"- Success: {scorer_data.success}")
            print(f"- Reason: {scorer_data.reason}")
            print(f"- Error: {scorer_data.error}")
            if scorer_data.verbose_logs:
                print(f"- Verbose Logs: {scorer_data.verbose_logs}")

        return False
    return True


if __name__ == "__main__":
    test_ac_scorer()
    test_ar_scorer()
    test_comparison_scorer()
    test_cp_scorer()
    test_cr_scorer()
    test_crelevancy_scorer()
    test_faithfulness_scorer()
    test_hallucination_scorer()
    test_instruction_adherence_scorer()
    test_summarization_scorer()
    
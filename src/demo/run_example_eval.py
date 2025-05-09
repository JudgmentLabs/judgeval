from judgeval import JudgmentClient
from judgeval.data import Example, Sequence
import random
from judgeval.scorers import AnswerRelevancyScorer, AnswerCorrectnessScorer

client = JudgmentClient()

for i in range(1):
    example = Example(
    input="What is the capital of France?",
    actual_output="Paris",
    expected_output="Paris",
    retrieval_context=["Paris is the capital of France.", "Paris is a city in France."],
    )
    example2 = Example(
        input="What is the capital of Kansas?",
        actual_output="Topeka",
        expected_output="Topeka",
        retrieval_context=["Topeka is the capital of Kansas.", "Topeka is a city in Kansas."],
    )
    example3 = Example(
        input="What is the capital of Texas?",
        actual_output="Austin",
        expected_output="Austin",
        retrieval_context=["Austin is the capital of Texas.", "Austin is a city in Texas."],
    )
    example4 = Example( 
        input="What is the capital of Nevada?",
        actual_output="Carson City",
        expected_output="Carson City",
        retrieval_context=["Carson City is the capital of Nevada.", "Carson City is a city in Nevada."],
    )
    example5 = Example(
        input="What is the capital of California?",
        actual_output="Sacramento",
        expected_output="Sacramento",
        retrieval_context=["Sacramento is the capital of California.", "Sacramento is a city in California."],
    )

    random_list = [example, example2, example3, example4, example5]
    num_to_select = random.randint(1, len(random_list))
    selected_elements = random.sample(random_list, num_to_select)

    results = client.run_evaluation(
        project_name="demo-35",
        eval_run_name=f"ALAN_TEST_RUN-{i + 1}",
        examples=selected_elements,
        scorers=[AnswerRelevancyScorer(threshold=0.5), AnswerCorrectnessScorer(threshold=0.5)],
        model="gpt-4o",
        append=True,
)

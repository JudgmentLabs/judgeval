"""
Implements the JudgmentClient to interact with the Judgment API.
"""
import os
from typing import Optional, List, Dict, Any, Union
import requests

from judgeval.constants import ROOT_API
from judgeval.data.datasets import EvalDataset, EvalDatasetClient
from judgeval.data import (
    ScorerData, 
    ScoringResult, 
    Example,
    CustomExample,
    Sequence,
)
from judgeval.scorers import (
    APIJudgmentScorer, 
    JudgevalScorer, 
    ClassifierScorer, 
)
from judgeval.evaluation_run import EvaluationRun
from judgeval.run_evaluation import (
    run_eval, 
    assert_test,
    run_sequence_eval
)
from judgeval.data.sequence_run import SequenceRun
from judgeval.judges import JudgevalJudge
from judgeval.constants import (
    JUDGMENT_EVAL_FETCH_API_URL, 
    JUDGMENT_EVAL_DELETE_API_URL, 
    JUDGMENT_EVAL_DELETE_PROJECT_API_URL,
    JUDGMENT_PROJECT_DELETE_API_URL,
    JUDGMENT_PROJECT_CREATE_API_URL
)
from judgeval.common.exceptions import JudgmentAPIError
from pydantic import BaseModel
from judgeval.rules import Rule

class EvalRunRequestBody(BaseModel):
    eval_name: str
    project_name: str
    judgment_api_key: str

class DeleteEvalRunRequestBody(BaseModel):
    eval_names: List[str]
    project_name: str
    judgment_api_key: str

class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class JudgmentClient(metaclass=SingletonMeta):
    def __init__(self, judgment_api_key: str = os.getenv("JUDGMENT_API_KEY"), organization_id: str = os.getenv("JUDGMENT_ORG_ID")):
        self.judgment_api_key = judgment_api_key
        self.organization_id = organization_id
        self.eval_dataset_client = EvalDatasetClient(judgment_api_key, organization_id)
        
        # Verify API key is valid
        result, response = self._validate_api_key()
        if not result:
            # May be bad to output their invalid API key...
            raise JudgmentAPIError(f"Issue with passed in Judgment API key: {response}")
        else:
            print(f"Successfully initialized JudgmentClient!")

    def a_run_evaluation(
        self, 
        examples: List[Example],
        scorers: List[Union[APIJudgmentScorer, JudgevalScorer]],
        model: Union[str, List[str], JudgevalJudge],
        aggregator: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        log_results: bool = True,
        project_name: str = "default_project",
        eval_run_name: str = "default_eval_run",
        override: bool = False,
        append: bool = False,
        ignore_errors: bool = True,
        rules: Optional[List[Rule]] = None
    ) -> List[ScoringResult]:
        return self.run_evaluation(
            examples=examples, 
            scorers=scorers, 
            model=model, 
            aggregator=aggregator, 
            metadata=metadata, 
            log_results=log_results, 
            project_name=project_name, 
            eval_run_name=eval_run_name, 
            override=override,
            append=append, 
            ignore_errors=ignore_errors, 
            rules=rules
        )

    def run_sequence_evaluation(
        self,
        sequences: List[Sequence],
        model: Union[str, List[str], JudgevalJudge],
        scorers: List[Union[APIJudgmentScorer, JudgevalScorer]],
        aggregator: Optional[str] = None,
        project_name: str = "default_project",
        eval_run_name: str = "default_eval_sequence",
        log_results: bool = True,
        append: bool = False,
        override: bool = False,
        ignore_errors: bool = True,
        rules: Optional[List[Rule]] = None
    ) -> List[ScoringResult]:
        try:
            def get_all_sequences(root: Sequence) -> List[Sequence]:
                all_sequences = [root]

                for item in root.items:
                    if isinstance(item, Sequence):
                        all_sequences.extend(get_all_sequences(item))

                return all_sequences

            def flatten_sequence_list(sequences: List[Sequence]) -> List[Sequence]:
                flattened = []
                for seq in sequences:
                    flattened.extend(get_all_sequences(seq))
                return flattened
            
            flattened_sequences = flatten_sequence_list(sequences)
            for sequence in flattened_sequences:
                sequence.scorers = scorers
                    
            sequence_run = SequenceRun(
                project_name=project_name,
                eval_name=eval_run_name,
                sequences=sequences,
                model=model,
                aggregator=aggregator,
                log_results=log_results,
                append=append,
                judgment_api_key=self.judgment_api_key,
                organization_id=self.organization_id
            )
            return run_sequence_eval(sequence_run, override, ignore_errors)
        except ValueError as e:
            raise ValueError(f"Please check your SequenceRun object, one or more fields are invalid: \n{str(e)}")
        except Exception as e:
            raise Exception(f"An unexpected error occurred during evaluation: {str(e)}")

    # TODO: Move this to Judgment as a metric
    def check_tool_order(self, trace, example):
        expected_tools = example.expected_tools
        current_tool_index = 0
        scorers_data = None
        for span in trace['entries']:
            if span['span_type'] == "tool":
                # Check for which object in the "inputs"
                # Ideally this is a separate field in the span
                var = span['inputs']['args']
                import re

                match = re.match(r"\((.*),\s*'(.*)'\)", var)
                if match:
                    obj_repr, city = match.groups()
                    tup = (obj_repr.strip(), city)
                
                obj_name, *args = tup
                
                if span['function'] != expected_tools[current_tool_index]['tool']:
                    scorers_data = ScorerData(
                        name='Tool-Order-Scorer',
                        threshold=1.0,
                        score=0.0,
                        success=False,
                        evaluation_model='gpt-4o-mini',
                        reason=f"Tool {span['function']} called out of order"
                    )
                
                if expected_tools[current_tool_index]['agent'] not in obj_name:
                    scorers_data = ScorerData(
                        name='Tool-Order-Scorer',
                        threshold=1.0,
                        score=0.0,
                        success=False,
                        evaluation_model='gpt-4o-mini',
                        reason=f"Agent {obj_name} called out of order"
                    )
                
                if args != list(expected_tools[current_tool_index]['params'].values()):
                    scorers_data = ScorerData(
                        name='Tool-Order-Scorer',
                        threshold=1.0,
                        score=0.0,
                        success=False,
                        evaluation_model='gpt-4o-mini',
                        reason=f"Args {args} called out of order"
                    )
                
                current_tool_index += 1
        
        if current_tool_index != len(expected_tools):
            raise ValueError(f"Not all tools were called, expected {len(expected_tools)} but got {current_tool_index}")
        
        if scorers_data is None:
            scorers_data = ScorerData(
                name='Tool-Order-Scorer',
                threshold=1.0,
                score=1.0,
                success=True,
                evaluation_model='gpt-4o-mini',
                reason="All tools called in order"
            )
        
        example.context = [str(expected_tools)]
        example.retrieval_context = [str(expected_tools)]
        return ScoringResult(
            success=scorers_data.success,
            scorers_data=[scorers_data],
            data_object=example
        )

    def run_evaluation(
        self, 
        examples: Union[List[Example], List[CustomExample]],
        scorers: List[Union[APIJudgmentScorer, JudgevalScorer]],
        model: Union[str, List[str], JudgevalJudge],
        aggregator: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        log_results: bool = True,
        project_name: str = "default_project",
        eval_run_name: str = "default_eval_run",
        override: bool = False,
        append: bool = False,
        ignore_errors: bool = True,
        async_execution: bool = False,
        rules: Optional[List[Rule]] = None
    ) -> List[ScoringResult]:
        """
        Executes an evaluation of `Example`s using one or more `Scorer`s
        
        Args:
            examples (Union[List[Example], List[CustomExample]]): The examples to evaluate
            scorers (List[Union[APIJudgmentScorer, JudgevalScorer]]): A list of scorers to use for evaluation
            model (Union[str, List[str], JudgevalJudge]): The model used as a judge when using LLM as a Judge
            aggregator (Optional[str]): The aggregator to use for evaluation if using Mixture of Judges
            metadata (Optional[Dict[str, Any]]): Additional metadata to include for this evaluation run
            log_results (bool): Whether to log the results to the Judgment API
            project_name (str): The name of the project the evaluation results belong to
            eval_run_name (str): A name for this evaluation run
            override (bool): Whether to override an existing evaluation run with the same name
            ignore_errors (bool): Whether to ignore errors during evaluation (safely handled)
            rules (Optional[List[Rule]]): Rules to evaluate against scoring results
            
        Returns:
            List[ScoringResult]: The results of the evaluation
        """
        if override and append:
            raise ValueError("Cannot set both override and append to True. Please choose one.")

        try:
            if rules and any(isinstance(scorer, JudgevalScorer) for scorer in scorers):
                raise ValueError("Cannot use Judgeval scorers (only API scorers) when using rules. Please either remove rules or use only APIJudgmentScorer types.")

            # 1. Run the traces
            import asyncio
            import openai
            import os
            from dotenv import load_dotenv
            from judgeval.tracer import Tracer, wrap

            # Initialize clients
            load_dotenv()
            client = wrap(openai.Client(api_key=os.getenv("OPENAI_API_KEY")))
            judgment = Tracer(
                api_key=self.judgment_api_key, 
                project_name="simple_trace_demo", 
            )


            class TravelAgent:
                @judgment.observe(span_type="tool")
                async def get_weather(self, city: str):
                    """Simulated weather tool call."""
                    weather_data = f"It is sunny and 72°F in {city}."
                    return weather_data

                @judgment.observe(span_type="tool")
                async def get_attractions(self, city: str):
                    """Simulated attractions tool call."""
                    attractions = [
                        "Eiffel Tower",
                        "Louvre Museum",
                        "Notre-Dame Cathedral",
                        "Arc de Triomphe"
                    ]
                    return attractions

                @judgment.observe(span_type="Research")
                async def gather_information(self, city: str):
                    """Gather all necessary travel information."""
                    weather = await self.get_weather(city)
                    attractions = await self.get_attractions(city)
                    
                    return {
                        "weather": weather,
                        "attractions": attractions
                    }

                @judgment.observe(span_type="function")
                async def create_travel_plan(self, research_data):
                    """Generate a travel itinerary using the researched data."""
                    prompt = f"""
                    Create a simple travel itinerary for Paris using this information:
                    
                    Weather: {research_data['weather']}
                    Attractions: {research_data['attractions']}
                    """
                    
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a travel planner. Create a simple itinerary."},
                            {"role": "user", "content": prompt}
                        ]
                    ).choices[0].message.content
                    
                    return response

                @judgment.observe(span_type="function")
                async def generate_simple_itinerary(self, query: str = "I want to plan a trip to Paris."):
                    """Main function to generate a travel itinerary."""
                    research_data = await self.gather_information(city="Paris")
                    itinerary = await self.create_travel_plan(research_data)
                    return itinerary
            
            agent1 = TravelAgent()
            # TODO: Make this clean
            itinerary, trace = asyncio.run(agent1.generate_simple_itinerary("I want to plan a trip to Paris."))
            traces = [trace]

            # 2. Loop through each trace
            results = []
            for trace in traces:
                # 3. Run the check
                print(f"Checking tool order for trace: {trace['trace_id']}")
                results = self.check_tool_order(trace, examples[0])
                print(f"Tool order result: {results=}")
            
            # 4. Log the results
            from judgeval.run_evaluation import log_evaluation_results, run_with_spinner
            from rich import print as rprint
            eval = EvaluationRun(
                log_results=log_results,
                append=append,
                project_name=project_name,
                eval_name=eval_run_name,
                examples=examples,
                scorers=scorers,
                model=model,
                aggregator=aggregator,
                metadata=metadata,
                judgment_api_key=self.judgment_api_key,
                rules=rules,
                organization_id=self.organization_id
            )
            
            # log_evaluation_results([results.model_dump()], eval)
            
            pretty_str = run_with_spinner("Logging Results: ", log_evaluation_results, [results.model_dump()], eval)
            rprint(pretty_str)
            
            print(f"{examples=}")
            
            # 5. Return the results (ie the results of the evaluation link)
            
            
            # 6. Temporary return
            return

            eval = EvaluationRun(
                log_results=log_results,
                append=append,
                project_name=project_name,
                eval_name=eval_run_name,
                examples=examples,
                scorers=scorers,
                model=model,
                aggregator=aggregator,
                metadata=metadata,
                judgment_api_key=self.judgment_api_key,
                rules=rules,
                organization_id=self.organization_id
            )
            return run_eval(eval, override, ignore_errors=ignore_errors, async_execution=async_execution)
        except ValueError as e:
            raise ValueError(f"Please check your EvaluationRun object, one or more fields are invalid: \n{str(e)}")
        except Exception as e:
            raise Exception(f"An unexpected error occurred during evaluation: {str(e)}")

    def create_dataset(self) -> EvalDataset:
        return self.eval_dataset_client.create_dataset()

    def push_dataset(self, alias: str, dataset: EvalDataset, project_name: str, overwrite: Optional[bool] = False) -> bool:
        """
        Uploads an `EvalDataset` to the Judgment platform for storage.

        Args:
            alias (str): The name to use for the dataset
            dataset (EvalDataset): The dataset to upload to Judgment
            overwrite (Optional[bool]): Whether to overwrite the dataset if it already exists

        Returns:
            bool: Whether the dataset was successfully uploaded
        """
        # Set judgment_api_key just in case it was not set
        dataset.judgment_api_key = self.judgment_api_key
        return self.eval_dataset_client.push(dataset, alias, project_name, overwrite)
    
    def append_example_dataset(self, alias: str, examples: List[Example], project_name: str) -> bool:
        """
        Appends an `EvalDataset` to the Judgment platform for storage.
        """
        return self.eval_dataset_client.append_examples(alias, examples, project_name)
    
    def append_sequence_dataset(self, alias: str, sequences: List[Sequence], project_name: str) -> bool:
        """
        Appends a `Sequence` to the Judgment platform for storage.
        """
        return self.eval_dataset_client.append_sequences(alias, sequences, project_name)
    
    def pull_dataset(self, alias: str, project_name: str) -> EvalDataset:
        """
        Retrieves a saved `EvalDataset` from the Judgment platform.

        Args:
            alias (str): The name of the dataset to retrieve

        Returns:
            EvalDataset: The retrieved dataset
        """
        return self.eval_dataset_client.pull(alias, project_name)

    def delete_dataset(self, alias: str, project_name: str) -> bool:
        """
        Deletes a saved `EvalDataset` from the Judgment platform.
        """
        return self.eval_dataset_client.delete(alias, project_name)
    
    def pull_project_dataset_stats(self, project_name: str) -> dict:
        """
        Retrieves all dataset stats from the Judgment platform for the project.

        Args:
            project_name (str): The name of the project to retrieve

        Returns:
            dict: The retrieved dataset stats
        """
        return self.eval_dataset_client.pull_project_dataset_stats(project_name)
    
    def insert_dataset(self, alias: str, examples: List[Example], project_name: str) -> bool:
        """
        Edits the dataset on Judgment platform by adding new examples
        """
        return self.eval_dataset_client.insert_dataset(alias, examples, project_name)
    
    # Maybe add option where you can pass in the EvaluationRun object and it will pull the eval results from the backend
    def pull_eval(self, project_name: str, eval_run_name: str) -> List[Dict[str, Union[str, List[ScoringResult]]]]:
        """Pull evaluation results from the server.

        Args:
            project_name (str): Name of the project
            eval_run_name (str): Name of the evaluation run

        Returns:
            Dict[str, Union[str, List[ScoringResult]]]: Dictionary containing:
                - id (str): The evaluation run ID
                - results (List[ScoringResult]): List of scoring results
        """
        eval_run_request_body = EvalRunRequestBody(project_name=project_name, 
                                                   eval_name=eval_run_name, 
                                                   judgment_api_key=self.judgment_api_key)
        eval_run = requests.post(JUDGMENT_EVAL_FETCH_API_URL,
                                 headers={
                                    "Content-Type": "application/json",
                                    "Authorization": f"Bearer {self.judgment_api_key}",
                                    "X-Organization-Id": self.organization_id
                                 },
                                 json=eval_run_request_body.model_dump(),
                                 verify=True)
        if eval_run.status_code != requests.codes.ok:
            raise ValueError(f"Error fetching eval results: {eval_run.json()}")

        return eval_run.json()
    
    def delete_eval(self, project_name: str, eval_run_names: List[str]) -> bool:
        """
        Deletes an evaluation from the server by project and run names.

        Args:
            project_name (str): Name of the project
            eval_run_names (List[str]): List of names of the evaluation runs

        Returns:
            bool: Whether the evaluation was successfully deleted
        """
        if not eval_run_names:
            raise ValueError("No evaluation run names provided")
        
        eval_run_request_body = DeleteEvalRunRequestBody(project_name=project_name, 
                                                   eval_names=eval_run_names, 
                                                   judgment_api_key=self.judgment_api_key)
        response = requests.delete(JUDGMENT_EVAL_DELETE_API_URL, 
                        json=eval_run_request_body.model_dump(),
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.judgment_api_key}",
                            "X-Organization-Id": self.organization_id
                        })
        if response.status_code == 404:
            raise ValueError(f"Eval results not found: {response.json()}")
        elif response.status_code == 500:
            raise ValueError(f"Error deleting eval results: {response.json()}")
        return bool(response.json())
    
    def delete_project_evals(self, project_name: str) -> bool:
        """
        Deletes all evaluations from the server for a given project.
        
        Args:
            project_name (str): Name of the project

        Returns:
            bool: Whether the evaluations were successfully deleted
        """
        response = requests.delete(JUDGMENT_EVAL_DELETE_PROJECT_API_URL, 
                        json={
                            "project_name": project_name,
                        },
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.judgment_api_key}",
                            "X-Organization-Id": self.organization_id
                        })
        if response.status_code != requests.codes.ok:
            raise ValueError(f"Error deleting eval results: {response.json()}")
        return response.json()
    
    def create_project(self, project_name: str) -> bool:
        """
        Creates a project on the server.
        """
        response = requests.post(JUDGMENT_PROJECT_CREATE_API_URL, 
                        json={
                            "project_name": project_name,
                        },
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.judgment_api_key}",
                            "X-Organization-Id": self.organization_id
                        })
        if response.status_code != requests.codes.ok:
            raise ValueError(f"Error creating project: {response.json()}")
        return response.json()
    
    def delete_project(self, project_name: str) -> bool:
        """
        Deletes a project from the server. Which also deletes all evaluations and traces associated with the project.
        """
        response = requests.delete(JUDGMENT_PROJECT_DELETE_API_URL, 
                        json={
                            "project_name": project_name,
                        },
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.judgment_api_key}",
                            "X-Organization-Id": self.organization_id
                        })
        if response.status_code != requests.codes.ok:
            raise ValueError(f"Error deleting project: {response.json()}")
        return response.json()
        
    def _validate_api_key(self):
        """
        Validates that the user api key is valid
        """
        response = requests.post(
            f"{ROOT_API}/validate_api_key/",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.judgment_api_key}",
            },
            json={},  # Empty body now
            verify=True
        )
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json().get("detail", "Error validating API key")

    def fetch_classifier_scorer(self, slug: str) -> ClassifierScorer:
        """
        Fetches a classifier scorer configuration from the Judgment API.

        Args:
            slug (str): Slug identifier of the custom scorer to fetch

        Returns:
            ClassifierScorer: The configured classifier scorer object

        Raises:
            JudgmentAPIError: If the scorer cannot be fetched or doesn't exist
        """
        request_body = {
            "slug": slug,
        }
        
        response = requests.post(
            f"{ROOT_API}/fetch_scorer/",
            json=request_body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.judgment_api_key}",
                "X-Organization-Id": self.organization_id
            },
            verify=True
        )
        
        if response.status_code == 500:
            raise JudgmentAPIError(f"The server is temporarily unavailable. Please try your request again in a few moments. Error details: {response.json().get('detail', '')}")
        elif response.status_code != 200:
            raise JudgmentAPIError(f"Failed to fetch classifier scorer '{slug}': {response.json().get('detail', '')}")
            
        scorer_config = response.json()
        created_at = scorer_config.pop("created_at")
        updated_at = scorer_config.pop("updated_at")
        
        try:
            return ClassifierScorer(**scorer_config)
        except Exception as e:
            raise JudgmentAPIError(f"Failed to create classifier scorer '{slug}' with config {scorer_config}: {str(e)}")

    def push_classifier_scorer(self, scorer: ClassifierScorer, slug: str = None) -> str:
        """
        Pushes a classifier scorer configuration to the Judgment API.

        Args:
            slug (str): Slug identifier for the scorer. If it exists, the scorer will be updated.
            scorer (ClassifierScorer): The classifier scorer to save

        Returns:
            str: The slug identifier of the saved scorer

        Raises:
            JudgmentAPIError: If there's an error saving the scorer
        """
        request_body = {
            "name": scorer.name,
            "conversation": scorer.conversation,
            "options": scorer.options,
            "slug": slug
        }
        
        response = requests.post(
            f"{ROOT_API}/save_scorer/",
            json=request_body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.judgment_api_key}",
                "X-Organization-Id": self.organization_id
            },
            verify=True
        )
        
        if response.status_code == 500:
            raise JudgmentAPIError(f"The server is temporarily unavailable. \
                                   Please try your request again in a few moments. \
                                   Error details: {response.json().get('detail', '')}")
        elif response.status_code != 200:
            raise JudgmentAPIError(f"Failed to save classifier scorer: {response.json().get('detail', '')}")
            
        return response.json()["slug"]
    
    def assert_test(
        self, 
        examples: List[Example],
        scorers: List[Union[APIJudgmentScorer, JudgevalScorer]],
        model: Union[str, List[str], JudgevalJudge],
        aggregator: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        log_results: bool = True,
        project_name: str = "default_project",
        eval_run_name: str = "default_eval_run",
        override: bool = False,
        rules: Optional[List[Rule]] = None
    ) -> None:
        """
        Asserts a test by running the evaluation and checking the results for success
        
        Args:
            examples (List[Example]): The examples to evaluate
            scorers (List[Union[APIJudgmentScorer, JudgevalScorer]]): A list of scorers to use for evaluation
            model (Union[str, List[str], JudgevalJudge]): The model used as a judge when using LLM as a Judge
            aggregator (Optional[str]): The aggregator to use for evaluation if using Mixture of Judges
            metadata (Optional[Dict[str, Any]]): Additional metadata to include for this evaluation run
            log_results (bool): Whether to log the results to the Judgment API
            project_name (str): The name of the project the evaluation results belong to
            eval_run_name (str): A name for this evaluation run
            override (bool): Whether to override an existing evaluation run with the same name
            rules (Optional[List[Rule]]): Rules to evaluate against scoring results
        """
        results = self.run_evaluation(
            examples=examples,
            scorers=scorers,
            model=model,
            aggregator=aggregator,
            metadata=metadata,
            log_results=log_results,
            project_name=project_name,
            eval_run_name=eval_run_name,
            override=override,
            rules=rules
        )
        
        assert_test(results)

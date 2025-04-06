"""
Implements the JudgmentClient to interact with the Judgment API.
"""
import os
from typing import Optional, List, Dict, Any, Union
import requests

from judgeval.constants import ROOT_API
from judgeval.data.datasets import EvalDataset, EvalDatasetClient
from judgeval.data import (
    ScoringResult, 
    Example,
)
from judgeval.scorers import (
    APIJudgmentScorer, 
    JudgevalScorer, 
    ClassifierScorer, 
    ScorerWrapper,
)
from judgeval.evaluation_run import EvaluationRun
from judgeval.run_evaluation import (
    run_eval, 
    assert_test
)
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
        scorers: List[Union[ScorerWrapper, JudgevalScorer]],
        model: Union[str, List[str], JudgevalJudge],
        aggregator: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        log_results: bool = True,
        project_name: str = "default_project",
        eval_run_name: str = "default_eval_run",
        override: bool = False,
        use_judgment: bool = True,
        ignore_errors: bool = True,
        rules: Optional[List[Rule]] = None
    ) -> List[ScoringResult]:
        return self.run_evaluation(examples, scorers, model, aggregator, metadata, log_results, project_name, eval_run_name, override, use_judgment, ignore_errors, True, rules)

    def run_evaluation(
        self, 
        examples: List[Example],
        scorers: List[Union[ScorerWrapper, JudgevalScorer]],
        model: Union[str, List[str], JudgevalJudge],
        aggregator: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        log_results: bool = True,
        project_name: str = "default_project",
        eval_run_name: str = "default_eval_run",
        override: bool = False,
        use_judgment: bool = True,
        ignore_errors: bool = True,
        async_execution: bool = False,
        rules: Optional[List[Rule]] = None
    ) -> List[ScoringResult]:
        """
        Executes an evaluation of `Example`s using one or more `Scorer`s
        
        Args:
            examples (List[Example]): The examples to evaluate
            scorers (List[Union[ScorerWrapper, JudgevalScorer]]): A list of scorers to use for evaluation
            model (Union[str, List[str], JudgevalJudge]): The model used as a judge when using LLM as a Judge
            aggregator (Optional[str]): The aggregator to use for evaluation if using Mixture of Judges
            metadata (Optional[Dict[str, Any]]): Additional metadata to include for this evaluation run
            log_results (bool): Whether to log the results to the Judgment API
            project_name (str): The name of the project the evaluation results belong to
            eval_run_name (str): A name for this evaluation run
            override (bool): Whether to override an existing evaluation run with the same name
            use_judgment (bool): Whether to use Judgment API for evaluation
            ignore_errors (bool): Whether to ignore errors during evaluation (safely handled)
            rules (Optional[List[Rule]]): Rules to evaluate against scoring results
            
        Returns:
            List[ScoringResult]: The results of the evaluation
        """
        try:
            # Load appropriate implementations for all scorers
            loaded_scorers: List[Union[JudgevalScorer, APIJudgmentScorer]] = []
            for scorer in scorers:
                try:
                    if isinstance(scorer, ScorerWrapper):
                        loaded_scorers.append(scorer.load_implementation(use_judgment=use_judgment))
                    else:
                        loaded_scorers.append(scorer)
                except Exception as e:
                    raise ValueError(f"Failed to load implementation for scorer {scorer}: {str(e)}")

            # Prevent using JudgevalScorer with rules - only APIJudgmentScorer allowed with rules
            if rules and any(isinstance(scorer, JudgevalScorer) for scorer in loaded_scorers):
                raise ValueError("Cannot use Judgeval scorers (only API scorers) when using rules. Please either remove rules or use only APIJudgmentScorer types.")

            # Convert ScorerWrapper in rules to their implementations
            loaded_rules = None
            if rules:
                loaded_rules = []
                for rule in rules:
                    try:
                        processed_conditions = []
                        for condition in rule.conditions:
                            # Convert metric if it's a ScorerWrapper
                            if isinstance(condition.metric, ScorerWrapper):
                                try:
                                    condition_copy = condition.model_copy()
                                    condition_copy.metric = condition.metric.load_implementation(use_judgment=use_judgment)
                                    processed_conditions.append(condition_copy)
                                except Exception as e:
                                    raise ValueError(f"Failed to convert ScorerWrapper to implementation in rule '{rule.name}', condition metric '{condition.metric}': {str(e)}")
                            else:
                                processed_conditions.append(condition)
                        
                        # Create new rule with processed conditions
                        new_rule = rule.model_copy()
                        new_rule.conditions = processed_conditions
                        loaded_rules.append(new_rule)
                    except Exception as e:
                        raise ValueError(f"Failed to process rule '{rule.name}': {str(e)}")

            eval = EvaluationRun(
                log_results=log_results,
                project_name=project_name,
                eval_name=eval_run_name,
                examples=examples,
                scorers=loaded_scorers,
                model=model,
                aggregator=aggregator,
                metadata=metadata,
                judgment_api_key=self.judgment_api_key,
                rules=loaded_rules,
                organization_id=self.organization_id
            )
            return run_eval(eval, override, ignore_errors=ignore_errors, async_execution=async_execution)
        except ValueError as e:
            raise ValueError(f"Please check your EvaluationRun object, one or more fields are invalid: \n{str(e)}")
        except Exception as e:
            raise Exception(f"An unexpected error occurred during evaluation: {str(e)}")
    
    def evaluate_dataset(
        self, 
        dataset: EvalDataset,
        scorers: List[Union[ScorerWrapper, JudgevalScorer]],
        model: Union[str, List[str], JudgevalJudge],
        aggregator: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        project_name: str = "",
        eval_run_name: str = "",
        log_results: bool = True,
        use_judgment: bool = True,
        rules: Optional[List[Rule]] = None
    ) -> List[ScoringResult]:
        """
        Executes an evaluation of a `EvalDataset` using one or more `Scorer`s
        
        Args:
            dataset (EvalDataset): The dataset containing examples to evaluate
            scorers (List[Union[ScorerWrapper, JudgevalScorer]]): A list of scorers to use for evaluation
            model (Union[str, List[str], JudgevalJudge]): The model used as a judge when using LLM as a Judge
            aggregator (Optional[str]): The aggregator to use for evaluation if using Mixture of Judges
            metadata (Optional[Dict[str, Any]]): Additional metadata to include for this evaluation run
            project_name (str): The name of the project the evaluation results belong to
            eval_run_name (str): A name for this evaluation run
            log_results (bool): Whether to log the results to the Judgment API
            use_judgment (bool): Whether to use Judgment API for evaluation
            rules (Optional[List[Rule]]): Rules to evaluate against scoring results
            
        Returns:
            List[ScoringResult]: The results of the evaluation
        """
        try:
            # Load appropriate implementations for all scorers
            loaded_scorers: List[Union[JudgevalScorer, APIJudgmentScorer]] = []
            for scorer in scorers:
                try:
                    if isinstance(scorer, ScorerWrapper):
                        loaded_scorers.append(scorer.load_implementation(use_judgment=use_judgment))
                    else:
                        loaded_scorers.append(scorer)
                except Exception as e:
                    raise ValueError(f"Failed to load implementation for scorer {scorer}: {str(e)}")

            # Prevent using JudgevalScorer with rules - only APIJudgmentScorer allowed with rules
            if rules and any(isinstance(scorer, JudgevalScorer) for scorer in loaded_scorers):
                raise ValueError("Cannot use Judgeval scorers (only API scorers) when using rules. Please either remove rules or use only APIJudgmentScorer types.")

            # Convert ScorerWrapper in rules to their implementations
            loaded_rules = None
            if rules:
                loaded_rules = []
                for rule in rules:
                    try:
                        processed_conditions = []
                        for condition in rule.conditions:
                            # Convert metric if it's a ScorerWrapper
                            if isinstance(condition.metric, ScorerWrapper):
                                try:
                                    condition_copy = condition.model_copy()
                                    condition_copy.metric = condition.metric.load_implementation(use_judgment=use_judgment)
                                    processed_conditions.append(condition_copy)
                                except Exception as e:
                                    raise ValueError(f"Failed to convert ScorerWrapper to implementation in rule '{rule.name}', condition metric '{condition.metric}': {str(e)}")
                            else:
                                processed_conditions.append(condition)
                        
                        # Create new rule with processed conditions
                        new_rule = rule.model_copy()
                        new_rule.conditions = processed_conditions
                        loaded_rules.append(new_rule)
                    except Exception as e:
                        raise ValueError(f"Failed to process rule '{rule.name}': {str(e)}")

            evaluation_run = EvaluationRun(
                log_results=log_results,
                project_name=project_name,
                eval_name=eval_run_name,
                examples=dataset.examples,
                scorers=loaded_scorers,
                model=model,
                aggregator=aggregator,
                metadata=metadata,
                judgment_api_key=self.judgment_api_key,
                rules=loaded_rules,
                organization_id=self.organization_id
            )
            return run_eval(evaluation_run)
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

        eval_run_result = [{}]
        for result in eval_run.json():
            result_id = result.get("id", "")
            result_data = result.get("result", dict())
            filtered_result = {k: v for k, v in result_data.items() if k in ScoringResult.__annotations__}
            eval_run_result[0]["id"] = result_id
            eval_run_result[0]["results"] = [ScoringResult(**filtered_result)]
        return eval_run_result
    
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

import type { Example } from "./data";
import { ExampleEvaluationRun } from "./data";
import { JUDGMENT_API_KEY, JUDGMENT_API_URL, JUDGMENT_ORG_ID } from "./env";
import { JudgmentApiClient } from "./internal/api";
import { ScorerData, ScoringResult } from "./internal/api/models";
import { LocalScorer } from "./scorers/local-scorer";
import { RemoteScorer } from "./scorers/remote-scorer";

/**
 * @deprecated Use v1 JudgmentClient instead. This class will be removed in a future version.
 * Import from 'judgeval/v1' instead.
 */
export class JudgmentClient {
  private readonly apiClient: JudgmentApiClient;

  constructor(
    apiKey: string | null = JUDGMENT_API_KEY,
    organizationId: string | null = JUDGMENT_ORG_ID,
  ) {
    if (apiKey === null || organizationId === null) {
      throw new Error("API key and organization ID are required");
    }

    this.apiClient = new JudgmentApiClient(
      JUDGMENT_API_URL,
      apiKey,
      organizationId,
    );
  }

  async runEvaluation(
    examples: Example[],
    scorers: RemoteScorer[],
    projectName: string,
    evalRunName: string,
    model?: string,
  ): Promise<ScoringResult[]>;
  async runEvaluation(
    examples: Example[],
    scorers: LocalScorer[],
    projectName: string,
    evalRunName: string,
    model?: string,
  ): Promise<ScoringResult[]>;
  async runEvaluation(
    examples: Example[],
    scorers: RemoteScorer[] | LocalScorer[],
    projectName: string,
    evalRunName: string,
    model?: string,
  ): Promise<ScoringResult[]> {
    if (examples.length === 0) {
      throw new Error("No examples provided");
    }

    const evalRun = new ExampleEvaluationRun({
      project_name: projectName,
      eval_name: evalRunName,
      examples,
      scorers,
      model,
    });

    const hasLocalScorers = Boolean(evalRun.local_scorers.length);
    const hasRemoteScorers = Boolean(evalRun.remote_scorers.length);

    if (hasLocalScorers && hasRemoteScorers) {
      throw new Error(
        "Cannot run both local and remote scorers at the same time",
      );
    }

    if (hasRemoteScorers) {
      return this.runRemoteScorers(evalRun);
    }

    return this.runLocalScorers(evalRun);
  }

  private async runRemoteScorers(
    evaluationRun: ExampleEvaluationRun,
  ): Promise<ScoringResult[]> {
    console.log("Running evaluation...");

    await this.apiClient.addToRunEvalQueueExamples(evaluationRun.toModel());

    const response = await this.pollForResults(
      evaluationRun.project_name,
      evaluationRun.id,
      evaluationRun.examples.length,
    );

    console.log(`View results at: ${response.ui_results_url}/example`);

    return response.results as unknown as ScoringResult[];
  }

  private async runLocalScorers(
    evaluationRun: ExampleEvaluationRun,
  ): Promise<ScoringResult[]> {
    console.log("Running local evaluation...");

    const results = await this.executeLocalScorers(evaluationRun);

    const response = await this.apiClient.logEvalResults({
      results,
      run: evaluationRun.toModel(),
    });

    console.log(`View results at: ${response.ui_results_url}/example`);

    return results;
  }

  private async executeLocalScorers(
    evaluationRun: ExampleEvaluationRun,
  ): Promise<ScoringResult[]> {
    const results: ScoringResult[] = [];

    for (const example of evaluationRun.examples) {
      const scorersData: ScorerData[] = [];

      for (const scorer of evaluationRun.local_scorers) {
        try {
          scorer.score = await scorer.scoreExample(example);
          scorer.success = scorer.successCheck();
        } catch (error) {
          scorer.error = error instanceof Error ? error.message : String(error);
          scorer.score = null;
          scorer.success = false;
        }

        scorersData.push({
          name: scorer.name,
          threshold: scorer.threshold,
          success: scorer.success,
          score: scorer.score,
          reason: scorer.reason,
          strict_mode: scorer.strictMode,
          evaluation_model: scorer.model,
          error: scorer.error,
          additional_metadata: scorer.metadata,
        });
      }

      results.push({
        success: scorersData.every((s) => s.success),
        scorers_data: scorersData,
        data_object: example,
      });
    }

    return results;
  }

  private async pollForResults(
    projectName: string,
    experimentRunId: string,
    expectedCount: number,
  ) {
    for (let i = 0; i < 60; i++) {
      const response = await this.apiClient.fetchExperimentRun({
        experiment_run_id: experimentRunId,
        project_name: projectName,
      });

      if (response.results?.length === expectedCount) {
        return response;
      }

      await new Promise((resolve) => setTimeout(resolve, 5000));
    }

    throw new Error("Evaluation timed out");
  }
}

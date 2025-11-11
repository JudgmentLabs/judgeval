import { ExampleEvaluationRun as ExampleEvaluationRunModel } from "../internal/api/models/ExampleEvaluationRun";
import { TraceEvaluationRun as TraceEvaluationRunModel } from "../internal/api/models/TraceEvaluationRun";
import {
  localScorerToBaseScorer,
  remoteScorerToScorerConfig,
} from "../scorers/adapters";
import { LocalScorer } from "../scorers/local-scorer";
import { RemoteScorer } from "../scorers/remote-scorer";
import { Example } from "./example";

export type ScorerInput = LocalScorer | RemoteScorer;

export class EvaluationRun {
  id: string;
  created_at: string;
  local_scorers: LocalScorer[] = [];
  remote_scorers: RemoteScorer[] = [];
  model?: string | null;

  constructor(params: { scorers?: ScorerInput[]; model?: string }) {
    const { scorers = [], model } = params;

    this.id = crypto.randomUUID();
    this.created_at = new Date().toISOString();
    this.model = model ?? null;

    for (const scorer of scorers) {
      if (scorer instanceof LocalScorer) {
        this.local_scorers.push(scorer);
      } else if (scorer instanceof RemoteScorer) {
        this.remote_scorers.push(scorer);
      }
    }
  }

  protected toBaseModel() {
    return {
      id: this.id,
      created_at: this.created_at,
      custom_scorers:
        this.local_scorers.length > 0
          ? this.local_scorers.map(localScorerToBaseScorer)
          : undefined,
      judgment_scorers:
        this.remote_scorers.length > 0
          ? this.remote_scorers.map(remoteScorerToScorerConfig)
          : undefined,
      model: this.model,
    };
  }
}

export class ExampleEvaluationRun
  extends EvaluationRun
  implements ExampleEvaluationRunModel
{
  project_name: string;
  eval_name: string;
  examples: Example[];
  trace_span_id?: string | null;
  trace_id?: string | null;

  constructor(params: {
    examples: Example[];
    scorers?: ScorerInput[];
    project_name: string;
    eval_name: string;
    model?: string;
    trace_span_id?: string | null;
    trace_id?: string | null;
  }) {
    const {
      examples,
      project_name,
      eval_name,
      trace_span_id,
      trace_id,
      ...baseParams
    } = params;

    super(baseParams);

    this.project_name = project_name;
    this.eval_name = eval_name;
    this.examples = examples;
    this.trace_span_id = trace_span_id ?? null;
    this.trace_id = trace_id ?? null;
  }

  toModel(): ExampleEvaluationRunModel {
    return {
      ...this.toBaseModel(),
      project_name: this.project_name,
      eval_name: this.eval_name,
      examples: this.examples,
      trace_span_id: this.trace_span_id,
      trace_id: this.trace_id,
    };
  }
}

export class TraceEvaluationRun
  extends EvaluationRun
  implements TraceEvaluationRunModel
{
  project_name: string;
  eval_name: string;
  trace_and_span_ids: [string, string][];
  is_offline?: boolean;

  constructor(params: {
    trace_and_span_ids: [string, string][];
    scorers?: ScorerInput[];
    project_name: string;
    eval_name: string;
    model?: string;
    is_offline?: boolean;
  }) {
    const {
      trace_and_span_ids,
      project_name,
      eval_name,
      is_offline,
      ...baseParams
    } = params;

    super(baseParams);

    this.project_name = project_name;
    this.eval_name = eval_name;
    this.trace_and_span_ids = trace_and_span_ids;
    this.is_offline = is_offline ?? false;
  }

  toModel(): TraceEvaluationRunModel {
    return {
      ...this.toBaseModel(),
      project_name: this.project_name,
      eval_name: this.eval_name,
      trace_and_span_ids: this.trace_and_span_ids,
      is_offline: this.is_offline,
    };
  }
}

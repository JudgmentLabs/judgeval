import type {
  Example as APIExample,
  OtelTraceSpan,
  ScoringResult as APIScoringResult,
} from "../../internal/api/models";
import { Example } from "./Example";
import { ScorerData } from "./ScorerData";

export interface ScoringResultConfig {
  success?: boolean;
  scorersData?: ScorerData[];
  name?: string;
  dataObject?: OtelTraceSpan | Example;
  traceId?: string;
  runDuration?: number;
  evaluationCost?: number;
}

export class ScoringResult {
  success?: boolean | null;
  scorersData?: ScorerData[];
  name?: string | null;
  dataObject?: OtelTraceSpan | Example | null;
  traceId?: string | null;
  runDuration?: number | null;
  evaluationCost?: number | null;

  constructor(config: ScoringResultConfig = {}) {
    this.success = config.success ?? null;
    this.scorersData = config.scorersData ?? [];
    this.name = config.name ?? null;
    this.dataObject = config.dataObject ?? null;
    this.traceId = config.traceId ?? null;
    this.runDuration = config.runDuration ?? null;
    this.evaluationCost = config.evaluationCost ?? null;
  }

  toModel(): APIScoringResult {
    const result: APIScoringResult = {
      success: this.success ?? false,
      scorers_data: (this.scorersData ?? []).map((s) => s.toModel()),
    };

    if (this.name !== undefined && this.name !== null) {
      result.name = this.name;
    }
    if (this.dataObject !== undefined && this.dataObject !== null) {
      if (this.dataObject instanceof Example) {
        result.data_object = this.dataObject.toModel();
      } else {
        result.data_object = this.dataObject as OtelTraceSpan | APIExample;
      }
    }
    if (this.traceId !== undefined && this.traceId !== null) {
      result.trace_id = this.traceId;
    }
    if (this.runDuration !== undefined && this.runDuration !== null) {
      result.run_duration = this.runDuration;
    }
    if (this.evaluationCost !== undefined && this.evaluationCost !== null) {
      result.evaluation_cost = this.evaluationCost;
    }

    return result;
  }
}

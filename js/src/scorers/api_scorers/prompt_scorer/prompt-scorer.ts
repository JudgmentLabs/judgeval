import { JUDGMENT_API_KEY, JUDGMENT_ORG_ID } from "../../../env";
import { BasePromptScorer } from "./base-prompt-scorer";
import { fetchPromptScorer, JudgmentAPIError } from "./prompt-scorer-utils";

export { fetchPromptScorer, JudgmentAPIError };

export class PromptScorer extends BasePromptScorer {
  constructor(
    name: string,
    prompt: string,
    threshold: number,
    model: string | null,
    options?: Record<string, number> | null,
    description?: string | null,
    judgmentApiKey: string = JUDGMENT_API_KEY ?? "",
    organizationId: string = JUDGMENT_ORG_ID ?? "",
  ) {
    super(
      "Prompt Scorer",
      name,
      prompt,
      threshold,
      model,
      options,
      description,
      judgmentApiKey,
      organizationId,
    );
  }

  protected getScoreType(): string {
    return "Prompt Scorer";
  }

  protected getIsTrace(): boolean {
    return false;
  }

  static async get(
    name: string,
    judgmentApiKey: string = JUDGMENT_API_KEY ?? "",
    organizationId: string = JUDGMENT_ORG_ID ?? "",
  ): Promise<PromptScorer> {
    const config = await fetchPromptScorer(
      name,
      judgmentApiKey,
      organizationId,
    );

    if (config.is_trace === true) {
      throw new JudgmentAPIError(
        400,
        `Scorer with name ${name} is a TracePromptScorer, not a PromptScorer`,
      );
    }

    return new PromptScorer(
      config.name,
      config.prompt,
      config.threshold,
      config.model ?? null,
      config.options,
      config.description,
      judgmentApiKey,
      organizationId,
    );
  }
}

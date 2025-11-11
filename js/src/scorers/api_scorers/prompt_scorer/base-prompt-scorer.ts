import { JUDGMENT_API_KEY, JUDGMENT_ORG_ID } from "../../../env";
import { RemoteScorer } from "../../remote-scorer";

export abstract class BasePromptScorer extends RemoteScorer {
  public prompt: string;
  public model: string | null;
  public options?: Record<string, number> | null;
  public description?: string | null;
  public judgmentApiKey: string;
  public organizationId: string;

  constructor(
    scoreType: string,
    name: string,
    prompt: string,
    threshold: number,
    model: string | null,
    options?: Record<string, number> | null,
    description?: string | null,
    judgmentApiKey: string = JUDGMENT_API_KEY ?? "",
    organizationId: string = JUDGMENT_ORG_ID ?? "",
  ) {
    super({
      scoreType,
      name,
      threshold,
      model,
      requiredParams: [],
      kwargs: {
        prompt,
        ...(options ? { options } : {}),
        ...(description ? { description } : {}),
      },
    });
    this.prompt = prompt;
    this.model = model;
    this.options = options;
    this.description = description;
    this.judgmentApiKey = judgmentApiKey;
    this.organizationId = organizationId;
  }

  protected abstract getScoreType(): string;
  protected abstract getIsTrace(): boolean;

  getThreshold(): number {
    return this.threshold;
  }

  getPrompt(): string {
    return this.prompt;
  }

  getOptions(): Record<string, number> | null {
    return this.options ? { ...this.options } : null;
  }

  getModel(): string | null {
    return this.model;
  }

  getDescription(): string | null | undefined {
    return this.description;
  }

  getName(): string {
    return this.name;
  }

  getConfig(): Record<string, unknown> {
    return {
      name: this.name,
      prompt: this.prompt,
      threshold: this.threshold,
      model: this.model,
      options: this.options,
      description: this.description,
    };
  }

  toString(): string {
    return `${this.constructor.name}(name=${this.name}, prompt=${
      this.prompt
    }, threshold=${this.threshold}, model=${this.model}, options=${JSON.stringify(this.options)}, description=${this.description})`;
  }
}

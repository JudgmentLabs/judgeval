import { JUDGEVAL_TRACER_INSTRUMENTING_MODULE_NAME } from "../constants";
import { JUDGMENT_API_KEY, JUDGMENT_API_URL, JUDGMENT_ORG_ID } from "../env";

/**
 * Configuration for the Judgment Tracer that controls how tracing and evaluation behave.
 *
 * This class encapsulates all configuration parameters needed to initialize a Tracer.
 *
 * @deprecated Use v1 TracerFactory instead. This class will be removed in a future version.
 * Import from 'judgeval/v1' instead.
 *
 * @example
 * ```typescript
 * const config = TracerConfiguration.builder()
 *   .projectName("my-project")
 *   .apiKey("your-api-key")
 *   .organizationId("your-org-id")
 *   .enableEvaluation(true)
 *   .resourceAttributes({
 *     "custom.service.version": "1.0.0",
 *     "custom.environment": "production"
 *   })
 *   .initialize(true)
 *   .build();
 *
 * const tracer = NodeTracer.createWithConfiguration(config);
 * // Tracer is ready to use immediately
 * ```
 */
export class TracerConfiguration {
  constructor(
    public readonly projectName: string,
    public readonly apiKey: string,
    public readonly organizationId: string,
    public readonly apiUrl: string,
    public readonly enableEvaluation: boolean,
    public readonly tracerName: string = JUDGEVAL_TRACER_INSTRUMENTING_MODULE_NAME,
    public readonly resourceAttributes: Record<string, unknown> = {},
    public readonly initialize = true,
  ) {}

  /**
   * Creates a default configuration with the given project name.
   *
   * This method uses default values from environment variables:
   * - API Key: JUDGMENT_API_KEY
   * - Organization ID: JUDGMENT_ORG_ID
   * - API URL: JUDGMENT_API_URL
   * - Evaluation: enabled
   *
   * @param projectName the name of the project
   * @returns a new TracerConfiguration with default values
   * @throws Error if project name is null or empty
   */
  public static createDefault(projectName: string): TracerConfiguration {
    return new TracerConfigurationBuilder().projectName(projectName).build();
  }

  public static builder(): TracerConfigurationBuilder {
    return new TracerConfigurationBuilder();
  }
}

/**
 * Builder for creating TracerConfiguration instances.
 *
 * @example
 * ```typescript
 * const config = TracerConfiguration.builder()
 *   .projectName("my-project")
 *   .apiKey("custom-api-key")
 *   .organizationId("custom-org-id")
 *   .apiUrl("https://custom-api.judgmentlabs.ai")
 *   .enableEvaluation(false)
 *   .resourceAttributes({
 *     "custom.service.version": "2.0.0",
 *     "custom.region": "us-west-2"
 *   })
 *   .initialize(false)
 *   .build();
 *
 * const tracer = NodeTracer.createWithConfiguration(config);
 * await tracer.initialize({
 *   instrumentations: []
 * });
 * ```
 */
export class TracerConfigurationBuilder {
  private _projectName?: string;
  private _apiKey: string | null = JUDGMENT_API_KEY;
  private _organizationId: string | null = JUDGMENT_ORG_ID;
  private _apiUrl: string = JUDGMENT_API_URL;
  private _enableEvaluation = true;
  private _tracerName: string = JUDGEVAL_TRACER_INSTRUMENTING_MODULE_NAME;
  private _resourceAttributes: Record<string, unknown> = {};
  private _initialize = true;

  public projectName(projectName: string): this {
    this._projectName = projectName;
    return this;
  }

  public apiKey(apiKey: string): this {
    this._apiKey = apiKey;
    return this;
  }

  public organizationId(organizationId: string): this {
    this._organizationId = organizationId;
    return this;
  }

  public apiUrl(apiUrl: string): this {
    this._apiUrl = apiUrl;
    return this;
  }

  public enableEvaluation(enableEvaluation: boolean): this {
    this._enableEvaluation = enableEvaluation;
    return this;
  }

  public tracerName(tracerName: string): this {
    this._tracerName = tracerName;
    return this;
  }

  public resourceAttributes(resourceAttributes: Record<string, unknown>): this {
    this._resourceAttributes = resourceAttributes;
    return this;
  }

  public initialize(initialize: boolean): this {
    this._initialize = initialize;
    return this;
  }

  public build(): TracerConfiguration {
    if (!this._projectName) {
      throw new Error("Project name is required");
    }

    if (!this._apiKey) {
      throw new Error("API key is required");
    }

    if (!this._organizationId) {
      throw new Error("Organization ID is required");
    }

    if (!this._apiUrl) {
      throw new Error("API URL is required");
    }

    return new TracerConfiguration(
      this._projectName,
      this._apiKey,
      this._organizationId,
      this._apiUrl,
      this._enableEvaluation,
      this._tracerName,
      this._resourceAttributes,
      this._initialize,
    );
  }
}

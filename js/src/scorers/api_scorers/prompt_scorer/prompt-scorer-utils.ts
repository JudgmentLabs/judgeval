import {
  JUDGMENT_API_KEY,
  JUDGMENT_API_URL,
  JUDGMENT_ORG_ID,
} from "../../../env";
import { JudgmentApiClient } from "../../../internal/api";

export class JudgmentAPIError extends Error {
  constructor(
    public statusCode: number,
    message: string,
  ) {
    super(message);
    this.name = "JudgmentAPIError";
  }
}

export async function fetchPromptScorer(
  name: string,
  judgmentApiKey: string = JUDGMENT_API_KEY ?? "",
  organizationId: string = JUDGMENT_ORG_ID ?? "",
) {
  if (!JUDGMENT_API_URL || !judgmentApiKey || !organizationId) {
    throw new Error("Missing required API credentials");
  }

  const client = new JudgmentApiClient(
    JUDGMENT_API_URL,
    judgmentApiKey,
    organizationId,
  );
  const response = await client.fetchScorers({ names: [name] });
  if (response.scorers.length === 0) {
    throw new JudgmentAPIError(404, `Scorer with name ${name} not found`);
  }
  return response.scorers[0];
}

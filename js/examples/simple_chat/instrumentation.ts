import { OpenAIInstrumentation } from "@opentelemetry/instrumentation-openai";
import { Judgeval, type NodeTracer } from "judgeval/v1";

export const client = Judgeval.create();

const initPromise = client.nodeTracer
  .create({
    projectName: "auto_instrumentation_example",
    enableEvaluation: true,
    enableMonitoring: true,
    instrumentations: [new OpenAIInstrumentation()],
  })
  .then((t: NodeTracer) => {
    return t;
  });

export async function getTracer(): Promise<NodeTracer> {
  return await initPromise;
}

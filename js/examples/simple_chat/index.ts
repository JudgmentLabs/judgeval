import { Example } from "judgeval/v1";
import OpenAI from "openai";
import { client, getTracer } from "./instrumentation";

function requireEnv(name: string): string {
  const value = process.env[name];
  if (!value) {
    throw new Error(`Environment variable ${name} is not set`);
  }
  return value;
}

const OPENAI_API_KEY = requireEnv("OPENAI_API_KEY");

const openai = new OpenAI({
  apiKey: OPENAI_API_KEY,
});

async function _chatWithUser(userMessage: string): Promise<string> {
  const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
    { role: "system", content: "You are a helpful assistant." },
    { role: "user", content: userMessage },
  ];

  const completion = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages,
  });

  const result = completion.choices[0].message.content || "";

  console.log(`User: ${userMessage}`);
  console.log(`Assistant: ${result}`);

  const tracer = await getTracer();

  tracer.asyncEvaluate(
    client.scorers.builtIn.answerRelevancy(),
    Example.create({
      properties: {
        input: "What is the capital of France?",
        actual_output: result,
      },
    })
  );

  return result;
}

(async () => {
  const tracer = await getTracer();
  const chatWithUser = tracer.observe(_chatWithUser);

  const result = await chatWithUser("What is the capital of France?");
  console.log(result);

  await new Promise((resolve) => setTimeout(resolve, 10000));
  await tracer.shutdown();
})();

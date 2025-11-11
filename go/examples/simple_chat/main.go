package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	v1 "github.com/JudgmentLabs/judgeval-go/v1"
	otelopenai "github.com/langwatch/langwatch/sdk-go/instrumentation/openai"
	"github.com/openai/openai-go"
	oaioption "github.com/openai/openai-go/option"
	"go.opentelemetry.io/otel/trace"
)

type ChatClient struct {
	client         openai.Client
	judgmentClient *v1.Judgeval
	tracer         *v1.Tracer
}

func NewChatClient(apiKey string) (*ChatClient, error) {
	client := openai.NewClient(
		oaioption.WithAPIKey(apiKey),
		oaioption.WithMiddleware(otelopenai.Middleware("default_project",
			otelopenai.WithCaptureInput(),
			otelopenai.WithCaptureOutput(),
		)),
	)

	return &ChatClient{
		client: client,
	}, nil
}

func (c *ChatClient) SetJudgeval(judgmentClient *v1.Judgeval, tracer *v1.Tracer) {
	c.judgmentClient = judgmentClient
	c.tracer = tracer
}

func (c *ChatClient) SendMessage(ctx context.Context, messages []openai.ChatCompletionMessageParamUnion) (string, error) {
	response, err := c.client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model:       openai.ChatModelGPT4,
		Messages:    messages,
		MaxTokens:   openai.Int(1000),
		Temperature: openai.Float(0.7),
	})
	if err != nil {
		return "", fmt.Errorf("chat completion failed: %w", err)
	}

	if len(response.Choices) == 0 {
		return "", fmt.Errorf("no response received from OpenAI")
	}

	return response.Choices[0].Message.Content, nil
}

func handleMessage(ctx context.Context, chatClient *ChatClient, userInput string, messages []openai.ChatCompletionMessageParamUnion, messageCount int) ([]openai.ChatCompletionMessageParamUnion, string, error) {
	var spanCtx context.Context
	var span trace.Span
	if chatClient.tracer != nil {
		spanCtx, span = chatClient.tracer.Span(ctx, "chat-message")
		defer span.End()
	} else {
		spanCtx = ctx
	}

	messages = append(messages, openai.SystemMessage("You are a helpful assistant. Echo whatever the user says."))

	messages = append(messages, openai.UserMessage(userInput))

	fmt.Print("Bot: ")
	botMessage, err := chatClient.SendMessage(spanCtx, messages)
	if err != nil {
		return messages[:len(messages)-1], "", err
	}

	fmt.Println(botMessage)
	messages = append(messages, openai.AssistantMessage(botMessage))

	if chatClient.tracer != nil && chatClient.judgmentClient != nil {
		chatClient.tracer.AsyncEvaluate(spanCtx, chatClient.judgmentClient.Scorers.BuiltIn.AnswerCorrectness(v1.AnswerCorrectnessScorerParams{
			Threshold: v1.Float(0.7),
		}), v1.NewExample(v1.ExampleParams{
			Name: v1.String(fmt.Sprintf("chat-message-%d", messageCount)),
			Properties: map[string]any{
				"input":           "You are a helpful assistant. Echo whatever the user says. Do not do anything else.",
				"actual_output":   botMessage,
				"expected_output": userInput,
			},
		}))
	}

	return messages, botMessage, nil
}

func main() {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Println("Error: OPENAI_API_KEY environment variable is required")
		fmt.Println("Please set it with: export OPENAI_API_KEY=your_api_key_here")
		os.Exit(1)
	}

	chatClient, err := NewChatClient(apiKey)
	if err != nil {
		fmt.Printf("Error: Failed to create chat client: %v\n", err)
		os.Exit(1)
	}

	var tracer *v1.Tracer
	var judgmentClient *v1.Judgeval
	if os.Getenv("JUDGMENT_API_URL") != "" && os.Getenv("JUDGMENT_API_KEY") != "" {
		client, err := v1.NewJudgeval(
			v1.WithAPIKey(os.Getenv("JUDGMENT_API_KEY")),
			v1.WithOrganizationID(os.Getenv("JUDGMENT_ORG_ID")),
		)
		if err != nil {
			fmt.Printf("Warning: Failed to create Judgment client: %v\n", err)
		} else {
			judgmentClient = client
			ctx := context.Background()
			tracer, err = client.Tracer.Create(ctx, v1.TracerCreateParams{
				ProjectName: "default_project",
			})
			if err != nil {
				fmt.Printf("Warning: Failed to initialize tracer: %v\n", err)
			} else {
				chatClient.SetJudgeval(judgmentClient, tracer)
				defer func() {
					shutdownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
					defer cancel()
					tracer.Shutdown(shutdownCtx)
				}()
			}
		}
	}

	ctx, span := tracer.StartSpan(context.Background(), "main")
	defer tracer.EndSpan(span)

	tracer.AsyncEvaluate(ctx, judgmentClient.Scorers.CustomScorer.Get("Helpfulness Scorer", "HelpfulnessScorer"), v1.NewExample(v1.ExampleParams{
		Properties: map[string]any{
			"question": "test",
			"answer":   "test",
		},
	}))

	fmt.Println("Simple Chat with OpenAI")
	fmt.Println("Type 'quit' or 'exit' to end the conversation")
	fmt.Println("Type 'clear' to clear conversation history")
	fmt.Println("----------------------------------------")

	var messages []openai.ChatCompletionMessageParamUnion
	scanner := bufio.NewScanner(os.Stdin)
	messageCount := 0

	for {
		fmt.Print("You: ")
		if !scanner.Scan() {
			break
		}

		userInput := strings.TrimSpace(scanner.Text())
		if userInput == "" {
			continue
		}

		if userInput == "quit" || userInput == "exit" {
			fmt.Println("Goodbye!")
			break
		}

		if userInput == "clear" {
			messages = nil
			fmt.Println("Conversation history cleared.")
			continue
		}

		messageCount++
		var err error
		messages, _, err = handleMessage(ctx, chatClient, userInput, messages, messageCount)
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			messageCount--
			continue
		}

		fmt.Println()
	}

	if err := scanner.Err(); err != nil {
		fmt.Printf("Error reading input: %v\n", err)
	}
}

// Package v1 provides the Judgment SDK v1 format for tracing, scoring, and evaluation.
//
// Basic usage:
//
//	client, err := v1.NewClient(
//		v1.WithAPIKey("your-api-key"),
//		v1.WithOrganizationID("your-org-id"),
//	)
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	tracer, err := client.Tracer.Create(ctx, v1.TracerCreateParams{
//		ProjectName: "my-project",
//		Initialize: v1.Bool(true),
//	})
//	if err != nil {
//		log.Fatal(err)
//	}
//	defer tracer.Shutdown(ctx)
//
//	scorer := client.Scorers.BuiltIn.Faithfulness(v1.FaithfulnessScorerParams{
//		Threshold: v1.Float(0.8),
//	})
//
//	example := v1.NewExample(v1.ExampleParams{
//		Name: v1.String("test-example"),
//		Properties: map[string]interface{}{
//			"input": "What is AI?",
//			"output": "Artificial Intelligence...",
//		},
//	})
package v1

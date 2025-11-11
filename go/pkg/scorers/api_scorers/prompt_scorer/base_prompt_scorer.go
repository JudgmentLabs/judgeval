package prompt_scorer

import (
	"fmt"
	"strconv"

	"github.com/JudgmentLabs/judgeval-go/pkg/data"
	"github.com/JudgmentLabs/judgeval-go/pkg/env"
	"github.com/JudgmentLabs/judgeval-go/pkg/internal/api"
	"github.com/JudgmentLabs/judgeval-go/pkg/internal/api/models"
	"github.com/JudgmentLabs/judgeval-go/pkg/scorers"
)

type BasePromptScorer struct {
	*scorers.APIScorer
	prompt         string
	options        map[string]float64
	judgmentAPIKey string
	organizationID string
}

type ScorerOptions struct {
	APIURL         string
	APIKey         string
	OrganizationID string
}

type ScorerOption func(*ScorerOptions)

func WithAPIKey(apiKey string) ScorerOption {
	return func(opts *ScorerOptions) {
		opts.APIKey = apiKey
	}
}

func WithOrganizationID(orgID string) ScorerOption {
	return func(opts *ScorerOptions) {
		opts.OrganizationID = orgID
	}
}

func NewBasePromptScorer(
	scoreType data.APIScorerType,
	name string,
	prompt string,
	threshold float64,
	options map[string]float64,
	judgmentAPIKey string,
	organizationID string,
) *BasePromptScorer {
	apiScorer := scorers.NewAPIScorer(scoreType, scorers.WithName(name), scorers.WithThreshold(threshold))

	return &BasePromptScorer{
		APIScorer:      apiScorer,
		prompt:         prompt,
		options:        options,
		judgmentAPIKey: judgmentAPIKey,
		organizationID: organizationID,
	}
}

func parseScorerOptions(options interface{}) map[string]float64 {
	result := make(map[string]float64)
	if options == nil {
		return result
	}

	if optionsMap, ok := options.(map[string]interface{}); ok {
		for k, v := range optionsMap {
			if num, ok := v.(float64); ok {
				result[k] = num
			} else if str, ok := v.(string); ok {
				if num, err := strconv.ParseFloat(str, 64); err == nil {
					result[k] = num
				}
			}
		}
	}
	return result
}

func ScorerExists(name, judgmentAPIKey, organizationID string) (bool, error) {
	client := api.NewClient(env.JudgmentAPIURL, judgmentAPIKey, organizationID)
	request := &models.ScorerExistsRequest{
		Name: name,
	}

	response, err := client.ScorerExists(request)
	if err != nil {
		return false, fmt.Errorf("failed to check if scorer exists: %v", err)
	}

	return response.Exists, nil
}

func FetchPromptScorer(name, judgmentAPIURL, judgmentAPIKey, organizationID string) (*models.PromptScorer, error) {
	client := api.NewClient(judgmentAPIURL, judgmentAPIKey, organizationID)
	request := &models.FetchPromptScorersRequest{
		Names: []string{name},
	}

	response, err := client.FetchScorers(request)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch prompt scorer '%s': %v", name, err)
	}

	if len(response.Scorers) == 0 {
		return nil, fmt.Errorf("failed to fetch prompt scorer '%s': not found", name)
	}

	return &response.Scorers[0], nil
}

func PushPromptScorer(
	name string,
	prompt string,
	threshold float64,
	options map[string]float64,
	judgmentAPIKey string,
	organizationID string,
	isTrace bool,
) (string, error) {
	client := api.NewClient(env.JudgmentAPIURL, judgmentAPIKey, organizationID)

	apiOptions := make(map[string]interface{})
	for k, v := range options {
		apiOptions[k] = v
	}

	request := &models.SavePromptScorerRequest{
		Name:      name,
		Prompt:    prompt,
		Threshold: threshold,
		Options:   apiOptions,
		IsTrace:   isTrace,
	}

	response, err := client.SaveScorer(request)
	if err != nil {
		return "", fmt.Errorf("failed to save prompt scorer: %v", err)
	}

	if response != nil {
		return response.ScorerResponse.Name, nil
	}
	return "", nil
}

func (bps *BasePromptScorer) GetPrompt() string {
	return bps.prompt
}

func (bps *BasePromptScorer) GetOptions() map[string]float64 {
	if bps.options == nil {
		return nil
	}
	result := make(map[string]float64)
	for k, v := range bps.options {
		result[k] = v
	}
	return result
}

func (bps *BasePromptScorer) GetScorerConfig() models.ScorerConfig {
	config := bps.APIScorer.GetScorerConfig()

	kwargs := make(map[string]interface{})
	kwargs["prompt"] = bps.GetPrompt()
	if bps.GetOptions() != nil {
		kwargs["options"] = bps.GetOptions()
	}

	if bps.APIScorer.AdditionalProperties != nil {
		for k, v := range bps.APIScorer.AdditionalProperties {
			kwargs[k] = v
		}
	}

	config.Kwargs = kwargs
	return config
}

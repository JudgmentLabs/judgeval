package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"

	"github.com/JudgmentLabs/judgeval-go/pkg/internal/api/models"
)

type Client struct {
	baseURL        string
	apiKey         string
	organizationID string
	httpClient     *http.Client
}

func NewClient(baseURL, apiKey, organizationID string) *Client {
	return &Client{
		baseURL:        baseURL,
		apiKey:         apiKey,
		organizationID: organizationID,
		httpClient:     &http.Client{},
	}
}

func (c *Client) buildURL(path string, queryParams map[string]string) string {
	u, _ := url.Parse(c.baseURL + path)
	if len(queryParams) > 0 {
		q := u.Query()
		for k, v := range queryParams {
			q.Set(k, v)
		}
		u.RawQuery = q.Encode()
	}
	return u.String()
}

func (c *Client) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("X-Organization-Id", c.organizationID)
}

func (c *Client) AddToRunEvalQueue(payload *models.ExampleEvaluationRun) (*models.EvalResults, error) {
	url := c.buildURL("/add_to_run_eval_queue/", nil)
	jsonPayload, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonPayload))
	if err != nil {
		return nil, err
	}
	c.setHeaders(req)
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HTTP Error: %d - %s", resp.StatusCode, string(body))
	}

	var result models.EvalResults
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	return &result, nil
}

func (c *Client) LogEvalResults(payload *models.EvalResults) (*models.EvalResults, error) {
	url := c.buildURL("/log_eval_results/", nil)
	jsonPayload, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonPayload))
	if err != nil {
		return nil, err
	}
	c.setHeaders(req)
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HTTP Error: %d - %s", resp.StatusCode, string(body))
	}

	var result models.EvalResults
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	return &result, nil
}

func (c *Client) FetchExperimentRun(payload *models.EvalResultsFetch) (*models.EvalResults, error) {
	url := c.buildURL("/fetch_experiment_run/", nil)
	jsonPayload, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonPayload))
	if err != nil {
		return nil, err
	}
	c.setHeaders(req)
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HTTP Error: %d - %s", resp.StatusCode, string(body))
	}

	var result models.EvalResults
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	return &result, nil
}

func (c *Client) ScorerExists(payload *models.ScorerExistsRequest) (*models.ScorerExistsResponse, error) {
	url := c.buildURL("/scorer_exists/", nil)
	jsonPayload, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonPayload))
	if err != nil {
		return nil, err
	}
	c.setHeaders(req)
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HTTP Error: %d - %s", resp.StatusCode, string(body))
	}

	var result models.ScorerExistsResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	return &result, nil
}

func (c *Client) SaveScorer(payload *models.SavePromptScorerRequest) (*models.SavePromptScorerResponse, error) {
	url := c.buildURL("/save_scorer/", nil)
	jsonPayload, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonPayload))
	if err != nil {
		return nil, err
	}
	c.setHeaders(req)
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HTTP Error: %d - %s", resp.StatusCode, string(body))
	}

	var result models.SavePromptScorerResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	return &result, nil
}

func (c *Client) FetchScorers(payload *models.FetchPromptScorersRequest) (*models.FetchPromptScorersResponse, error) {
	url := c.buildURL("/fetch_scorers/", nil)
	jsonPayload, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonPayload))
	if err != nil {
		return nil, err
	}
	c.setHeaders(req)
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HTTP Error: %d - %s", resp.StatusCode, string(body))
	}

	var result models.FetchPromptScorersResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	return &result, nil
}

func (c *Client) ProjectsResolve(payload *models.ResolveProjectNameRequest) (*models.ResolveProjectNameResponse, error) {
	url := c.buildURL("/projects/resolve/", nil)
	jsonPayload, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonPayload))
	if err != nil {
		return nil, err
	}
	c.setHeaders(req)
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HTTP Error: %d - %s", resp.StatusCode, string(body))
	}

	var result models.ResolveProjectNameResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	return &result, nil
}

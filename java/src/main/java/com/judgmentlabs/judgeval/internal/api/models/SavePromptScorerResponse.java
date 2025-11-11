package com.judgmentlabs.judgeval.internal.api.models;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

import com.fasterxml.jackson.annotation.JsonAnyGetter;
import com.fasterxml.jackson.annotation.JsonAnySetter;
import com.fasterxml.jackson.annotation.JsonProperty;

public class SavePromptScorerResponse {
    @JsonProperty("scorer_response")
    private PromptScorer        scorerResponse;

    private Map<String, Object> additionalProperties = new HashMap<>();

    @JsonAnyGetter
    public Map<String, Object> getAdditionalProperties() {
        return additionalProperties;
    }

    @JsonAnySetter
    public void setAdditionalProperty(String name, Object value) {
        additionalProperties.put(name, value);
    }

    public PromptScorer getScorerResponse() {
        return scorerResponse;
    }

    public void setScorerResponse(PromptScorer scorerResponse) {
        this.scorerResponse = scorerResponse;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null || getClass() != obj.getClass())
            return false;
        SavePromptScorerResponse other = (SavePromptScorerResponse) obj;
        return Objects.equals(scorerResponse, other.scorerResponse)
                && Objects.equals(additionalProperties, other.additionalProperties);
    }

    @Override
    public int hashCode() {
        return Objects.hash(scorerResponse, Objects.hashCode(additionalProperties));
    }
}
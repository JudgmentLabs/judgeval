package com.judgmentlabs.judgeval.internal.api.models;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import com.fasterxml.jackson.annotation.JsonAnyGetter;
import com.fasterxml.jackson.annotation.JsonAnySetter;
import com.fasterxml.jackson.annotation.JsonProperty;

public class FetchPromptScorersResponse {
    @JsonProperty("scorers")
    private List<PromptScorer>  scorers;

    private Map<String, Object> additionalProperties = new HashMap<>();

    @JsonAnyGetter
    public Map<String, Object> getAdditionalProperties() {
        return additionalProperties;
    }

    @JsonAnySetter
    public void setAdditionalProperty(String name, Object value) {
        additionalProperties.put(name, value);
    }

    public List<PromptScorer> getScorers() {
        return scorers;
    }

    public void setScorers(List<PromptScorer> scorers) {
        this.scorers = scorers;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null || getClass() != obj.getClass())
            return false;
        FetchPromptScorersResponse other = (FetchPromptScorersResponse) obj;
        return Objects.equals(scorers, other.scorers)
                && Objects.equals(additionalProperties, other.additionalProperties);
    }

    @Override
    public int hashCode() {
        return Objects.hash(scorers, Objects.hashCode(additionalProperties));
    }
}
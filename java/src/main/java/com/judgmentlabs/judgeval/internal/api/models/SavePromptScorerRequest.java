package com.judgmentlabs.judgeval.internal.api.models;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

import com.fasterxml.jackson.annotation.JsonAnyGetter;
import com.fasterxml.jackson.annotation.JsonAnySetter;
import com.fasterxml.jackson.annotation.JsonProperty;

public class SavePromptScorerRequest {
    @JsonProperty("name")
    private String              name;
    @JsonProperty("prompt")
    private String              prompt;
    @JsonProperty("threshold")
    private Double              threshold;
    @JsonProperty("model")
    private String              model;
    @JsonProperty("is_trace")
    private Boolean             isTrace;
    @JsonProperty("options")
    private Object              options;
    @JsonProperty("description")
    private String              description;

    private Map<String, Object> additionalProperties = new HashMap<>();

    @JsonAnyGetter
    public Map<String, Object> getAdditionalProperties() {
        return additionalProperties;
    }

    @JsonAnySetter
    public void setAdditionalProperty(String name, Object value) {
        additionalProperties.put(name, value);
    }

    public String getName() {
        return name;
    }

    public String getPrompt() {
        return prompt;
    }

    public Double getThreshold() {
        return threshold;
    }

    public String getModel() {
        return model;
    }

    public Boolean getIsTrace() {
        return isTrace;
    }

    public Object getOptions() {
        return options;
    }

    public String getDescription() {
        return description;
    }

    public void setName(String name) {
        this.name = name;
    }

    public void setPrompt(String prompt) {
        this.prompt = prompt;
    }

    public void setThreshold(Double threshold) {
        this.threshold = threshold;
    }

    public void setModel(String model) {
        this.model = model;
    }

    public void setIsTrace(Boolean isTrace) {
        this.isTrace = isTrace;
    }

    public void setOptions(Object options) {
        this.options = options;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null || getClass() != obj.getClass())
            return false;
        SavePromptScorerRequest other = (SavePromptScorerRequest) obj;
        return Objects.equals(name, other.name) && Objects.equals(prompt, other.prompt)
                && Objects.equals(threshold, other.threshold) && Objects.equals(model, other.model)
                && Objects.equals(isTrace, other.isTrace) && Objects.equals(options, other.options)
                && Objects.equals(description, other.description)
                && Objects.equals(additionalProperties, other.additionalProperties);
    }

    @Override
    public int hashCode() {
        return Objects.hash(name, prompt, threshold, model, isTrace, options, description,
                Objects.hashCode(additionalProperties));
    }
}
package com.judgmentlabs.judgeval.internal.api.models;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

import com.fasterxml.jackson.annotation.JsonAnyGetter;
import com.fasterxml.jackson.annotation.JsonAnySetter;
import com.fasterxml.jackson.annotation.JsonProperty;

public class ScorerData {
    @JsonProperty("id")
    private String              id;
    @JsonProperty("name")
    private String              name;
    @JsonProperty("threshold")
    private Double              threshold;
    @JsonProperty("success")
    private Boolean             success;
    @JsonProperty("score")
    private Double              score;
    @JsonProperty("reason")
    private String              reason;
    @JsonProperty("strict_mode")
    private Boolean             strictMode;
    @JsonProperty("evaluation_model")
    private String              evaluationModel;
    @JsonProperty("error")
    private String              error;
    @JsonProperty("additional_metadata")
    private Object              additionalMetadata;

    private Map<String, Object> additionalProperties = new HashMap<>();

    @JsonAnyGetter
    public Map<String, Object> getAdditionalProperties() {
        return additionalProperties;
    }

    @JsonAnySetter
    public void setAdditionalProperty(String name, Object value) {
        additionalProperties.put(name, value);
    }

    public String getId() {
        return id;
    }

    public String getName() {
        return name;
    }

    public Double getThreshold() {
        return threshold;
    }

    public Boolean getSuccess() {
        return success;
    }

    public Double getScore() {
        return score;
    }

    public String getReason() {
        return reason;
    }

    public Boolean getStrictMode() {
        return strictMode;
    }

    public String getEvaluationModel() {
        return evaluationModel;
    }

    public String getError() {
        return error;
    }

    public Object getAdditionalMetadata() {
        return additionalMetadata;
    }

    public void setId(String id) {
        this.id = id;
    }

    public void setName(String name) {
        this.name = name;
    }

    public void setThreshold(Double threshold) {
        this.threshold = threshold;
    }

    public void setSuccess(Boolean success) {
        this.success = success;
    }

    public void setScore(Double score) {
        this.score = score;
    }

    public void setReason(String reason) {
        this.reason = reason;
    }

    public void setStrictMode(Boolean strictMode) {
        this.strictMode = strictMode;
    }

    public void setEvaluationModel(String evaluationModel) {
        this.evaluationModel = evaluationModel;
    }

    public void setError(String error) {
        this.error = error;
    }

    public void setAdditionalMetadata(Object additionalMetadata) {
        this.additionalMetadata = additionalMetadata;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null || getClass() != obj.getClass())
            return false;
        ScorerData other = (ScorerData) obj;
        return Objects.equals(id, other.id) && Objects.equals(name, other.name)
                && Objects.equals(threshold, other.threshold) && Objects.equals(success, other.success)
                && Objects.equals(score, other.score) && Objects.equals(reason, other.reason)
                && Objects.equals(strictMode, other.strictMode)
                && Objects.equals(evaluationModel, other.evaluationModel) && Objects.equals(error, other.error)
                && Objects.equals(additionalMetadata, other.additionalMetadata)
                && Objects.equals(additionalProperties, other.additionalProperties);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id, name, threshold, success, score, reason, strictMode, evaluationModel, error,
                additionalMetadata, Objects.hashCode(additionalProperties));
    }
}
package com.judgmentlabs.judgeval.internal.api.models;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

import com.fasterxml.jackson.annotation.JsonAnyGetter;
import com.fasterxml.jackson.annotation.JsonAnySetter;
import com.fasterxml.jackson.annotation.JsonProperty;

public class BaseScorer {
    @JsonProperty("score_type")
    private String              scoreType;
    @JsonProperty("threshold")
    private Double              threshold;
    @JsonProperty("name")
    private String              name;
    @JsonProperty("class_name")
    private String              className;
    @JsonProperty("score")
    private Double              score;
    @JsonProperty("score_breakdown")
    private Object              scoreBreakdown;
    @JsonProperty("reason")
    private String              reason;
    @JsonProperty("using_native_model")
    private Boolean             usingNativeModel;
    @JsonProperty("success")
    private Boolean             success;
    @JsonProperty("model")
    private String              model;
    @JsonProperty("model_client")
    private Object              modelClient;
    @JsonProperty("strict_mode")
    private Boolean             strictMode;
    @JsonProperty("error")
    private String              error;
    @JsonProperty("additional_metadata")
    private Object              additionalMetadata;
    @JsonProperty("user")
    private String              user;
    @JsonProperty("server_hosted")
    private Boolean             serverHosted;

    private Map<String, Object> additionalProperties = new HashMap<>();

    @JsonAnyGetter
    public Map<String, Object> getAdditionalProperties() {
        return additionalProperties;
    }

    @JsonAnySetter
    public void setAdditionalProperty(String name, Object value) {
        additionalProperties.put(name, value);
    }

    public String getScoreType() {
        return scoreType;
    }

    public Double getThreshold() {
        return threshold;
    }

    public String getName() {
        return name;
    }

    public String getClassName() {
        return className;
    }

    public Double getScore() {
        return score;
    }

    public Object getScoreBreakdown() {
        return scoreBreakdown;
    }

    public String getReason() {
        return reason;
    }

    public Boolean getUsingNativeModel() {
        return usingNativeModel;
    }

    public Boolean getSuccess() {
        return success;
    }

    public String getModel() {
        return model;
    }

    public Object getModelClient() {
        return modelClient;
    }

    public Boolean getStrictMode() {
        return strictMode;
    }

    public String getError() {
        return error;
    }

    public Object getAdditionalMetadata() {
        return additionalMetadata;
    }

    public String getUser() {
        return user;
    }

    public Boolean getServerHosted() {
        return serverHosted;
    }

    public void setScoreType(String scoreType) {
        this.scoreType = scoreType;
    }

    public void setThreshold(Double threshold) {
        this.threshold = threshold;
    }

    public void setName(String name) {
        this.name = name;
    }

    public void setClassName(String className) {
        this.className = className;
    }

    public void setScore(Double score) {
        this.score = score;
    }

    public void setScoreBreakdown(Object scoreBreakdown) {
        this.scoreBreakdown = scoreBreakdown;
    }

    public void setReason(String reason) {
        this.reason = reason;
    }

    public void setUsingNativeModel(Boolean usingNativeModel) {
        this.usingNativeModel = usingNativeModel;
    }

    public void setSuccess(Boolean success) {
        this.success = success;
    }

    public void setModel(String model) {
        this.model = model;
    }

    public void setModelClient(Object modelClient) {
        this.modelClient = modelClient;
    }

    public void setStrictMode(Boolean strictMode) {
        this.strictMode = strictMode;
    }

    public void setError(String error) {
        this.error = error;
    }

    public void setAdditionalMetadata(Object additionalMetadata) {
        this.additionalMetadata = additionalMetadata;
    }

    public void setUser(String user) {
        this.user = user;
    }

    public void setServerHosted(Boolean serverHosted) {
        this.serverHosted = serverHosted;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null || getClass() != obj.getClass())
            return false;
        BaseScorer other = (BaseScorer) obj;
        return Objects.equals(scoreType, other.scoreType) && Objects.equals(threshold, other.threshold)
                && Objects.equals(name, other.name) && Objects.equals(className, other.className)
                && Objects.equals(score, other.score) && Objects.equals(scoreBreakdown, other.scoreBreakdown)
                && Objects.equals(reason, other.reason) && Objects.equals(usingNativeModel, other.usingNativeModel)
                && Objects.equals(success, other.success) && Objects.equals(model, other.model)
                && Objects.equals(modelClient, other.modelClient) && Objects.equals(strictMode, other.strictMode)
                && Objects.equals(error, other.error) && Objects.equals(additionalMetadata, other.additionalMetadata)
                && Objects.equals(user, other.user) && Objects.equals(serverHosted, other.serverHosted)
                && Objects.equals(additionalProperties, other.additionalProperties);
    }

    @Override
    public int hashCode() {
        return Objects.hash(scoreType, threshold, name, className, score, scoreBreakdown, reason, usingNativeModel,
                success, model, modelClient, strictMode, error, additionalMetadata, user, serverHosted,
                Objects.hashCode(additionalProperties));
    }
}
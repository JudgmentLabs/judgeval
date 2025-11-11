package com.judgmentlabs.judgeval.internal.api.models;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import com.fasterxml.jackson.annotation.JsonAnyGetter;
import com.fasterxml.jackson.annotation.JsonAnySetter;
import com.fasterxml.jackson.annotation.JsonProperty;

public class TraceEvaluationRun {
    @JsonProperty("id")
    private String              id;
    @JsonProperty("project_name")
    private String              projectName;
    @JsonProperty("eval_name")
    private String              evalName;
    @JsonProperty("custom_scorers")
    private List<BaseScorer>    customScorers;
    @JsonProperty("judgment_scorers")
    private List<ScorerConfig>  judgmentScorers;
    @JsonProperty("model")
    private String              model;
    @JsonProperty("created_at")
    private String              createdAt;
    @JsonProperty("trace_and_span_ids")
    private List<List<Object>>  traceAndSpanIds;
    @JsonProperty("is_offline")
    private Boolean             isOffline;
    @JsonProperty("is_bucket_run")
    private Boolean             isBucketRun;

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

    public String getProjectName() {
        return projectName;
    }

    public String getEvalName() {
        return evalName;
    }

    public List<BaseScorer> getCustomScorers() {
        return customScorers;
    }

    public List<ScorerConfig> getJudgmentScorers() {
        return judgmentScorers;
    }

    public String getModel() {
        return model;
    }

    public String getCreatedAt() {
        return createdAt;
    }

    public List<List<Object>> getTraceAndSpanIds() {
        return traceAndSpanIds;
    }

    public Boolean getIsOffline() {
        return isOffline;
    }

    public Boolean getIsBucketRun() {
        return isBucketRun;
    }

    public void setId(String id) {
        this.id = id;
    }

    public void setProjectName(String projectName) {
        this.projectName = projectName;
    }

    public void setEvalName(String evalName) {
        this.evalName = evalName;
    }

    public void setCustomScorers(List<BaseScorer> customScorers) {
        this.customScorers = customScorers;
    }

    public void setJudgmentScorers(List<ScorerConfig> judgmentScorers) {
        this.judgmentScorers = judgmentScorers;
    }

    public void setModel(String model) {
        this.model = model;
    }

    public void setCreatedAt(String createdAt) {
        this.createdAt = createdAt;
    }

    public void setTraceAndSpanIds(List<List<Object>> traceAndSpanIds) {
        this.traceAndSpanIds = traceAndSpanIds;
    }

    public void setIsOffline(Boolean isOffline) {
        this.isOffline = isOffline;
    }

    public void setIsBucketRun(Boolean isBucketRun) {
        this.isBucketRun = isBucketRun;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null || getClass() != obj.getClass())
            return false;
        TraceEvaluationRun other = (TraceEvaluationRun) obj;
        return Objects.equals(id, other.id) && Objects.equals(projectName, other.projectName)
                && Objects.equals(evalName, other.evalName) && Objects.equals(customScorers, other.customScorers)
                && Objects.equals(judgmentScorers, other.judgmentScorers) && Objects.equals(model, other.model)
                && Objects.equals(createdAt, other.createdAt) && Objects.equals(traceAndSpanIds, other.traceAndSpanIds)
                && Objects.equals(isOffline, other.isOffline) && Objects.equals(isBucketRun, other.isBucketRun)
                && Objects.equals(additionalProperties, other.additionalProperties);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id, projectName, evalName, customScorers, judgmentScorers, model, createdAt,
                traceAndSpanIds, isOffline, isBucketRun, Objects.hashCode(additionalProperties));
    }
}
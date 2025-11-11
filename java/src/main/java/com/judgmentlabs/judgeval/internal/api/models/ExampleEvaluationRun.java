package com.judgmentlabs.judgeval.internal.api.models;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import com.fasterxml.jackson.annotation.JsonAnyGetter;
import com.fasterxml.jackson.annotation.JsonAnySetter;
import com.fasterxml.jackson.annotation.JsonProperty;

public class ExampleEvaluationRun {
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
    @JsonProperty("examples")
    private List<Example>       examples;
    @JsonProperty("trace_span_id")
    private String              traceSpanId;
    @JsonProperty("trace_id")
    private String              traceId;

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

    public List<Example> getExamples() {
        return examples;
    }

    public String getTraceSpanId() {
        return traceSpanId;
    }

    public String getTraceId() {
        return traceId;
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

    public void setExamples(List<Example> examples) {
        this.examples = examples;
    }

    public void setTraceSpanId(String traceSpanId) {
        this.traceSpanId = traceSpanId;
    }

    public void setTraceId(String traceId) {
        this.traceId = traceId;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null || getClass() != obj.getClass())
            return false;
        ExampleEvaluationRun other = (ExampleEvaluationRun) obj;
        return Objects.equals(id, other.id) && Objects.equals(projectName, other.projectName)
                && Objects.equals(evalName, other.evalName) && Objects.equals(customScorers, other.customScorers)
                && Objects.equals(judgmentScorers, other.judgmentScorers) && Objects.equals(model, other.model)
                && Objects.equals(createdAt, other.createdAt) && Objects.equals(examples, other.examples)
                && Objects.equals(traceSpanId, other.traceSpanId) && Objects.equals(traceId, other.traceId)
                && Objects.equals(additionalProperties, other.additionalProperties);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id, projectName, evalName, customScorers, judgmentScorers, model, createdAt, examples,
                traceSpanId, traceId, Objects.hashCode(additionalProperties));
    }
}
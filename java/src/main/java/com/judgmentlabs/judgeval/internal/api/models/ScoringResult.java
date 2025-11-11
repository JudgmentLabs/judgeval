package com.judgmentlabs.judgeval.internal.api.models;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import com.fasterxml.jackson.annotation.JsonAnyGetter;
import com.fasterxml.jackson.annotation.JsonAnySetter;
import com.fasterxml.jackson.annotation.JsonProperty;

public class ScoringResult {
    @JsonProperty("success")
    private Boolean             success;
    @JsonProperty("scorers_data")
    private List<ScorerData>    scorersData;
    @JsonProperty("name")
    private String              name;
    @JsonProperty("data_object")
    private Object              dataObject;
    @JsonProperty("trace_id")
    private String              traceId;
    @JsonProperty("run_duration")
    private Double              runDuration;
    @JsonProperty("evaluation_cost")
    private Double              evaluationCost;

    private Map<String, Object> additionalProperties = new HashMap<>();

    @JsonAnyGetter
    public Map<String, Object> getAdditionalProperties() {
        return additionalProperties;
    }

    @JsonAnySetter
    public void setAdditionalProperty(String name, Object value) {
        additionalProperties.put(name, value);
    }

    public Boolean getSuccess() {
        return success;
    }

    public List<ScorerData> getScorersData() {
        return scorersData;
    }

    public String getName() {
        return name;
    }

    public Object getDataObject() {
        return dataObject;
    }

    public String getTraceId() {
        return traceId;
    }

    public Double getRunDuration() {
        return runDuration;
    }

    public Double getEvaluationCost() {
        return evaluationCost;
    }

    public void setSuccess(Boolean success) {
        this.success = success;
    }

    public void setScorersData(List<ScorerData> scorersData) {
        this.scorersData = scorersData;
    }

    public void setName(String name) {
        this.name = name;
    }

    public void setDataObject(Object dataObject) {
        this.dataObject = dataObject;
    }

    public void setTraceId(String traceId) {
        this.traceId = traceId;
    }

    public void setRunDuration(Double runDuration) {
        this.runDuration = runDuration;
    }

    public void setEvaluationCost(Double evaluationCost) {
        this.evaluationCost = evaluationCost;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null || getClass() != obj.getClass())
            return false;
        ScoringResult other = (ScoringResult) obj;
        return Objects.equals(success, other.success) && Objects.equals(scorersData, other.scorersData)
                && Objects.equals(name, other.name) && Objects.equals(dataObject, other.dataObject)
                && Objects.equals(traceId, other.traceId) && Objects.equals(runDuration, other.runDuration)
                && Objects.equals(evaluationCost, other.evaluationCost)
                && Objects.equals(additionalProperties, other.additionalProperties);
    }

    @Override
    public int hashCode() {
        return Objects.hash(success, scorersData, name, dataObject, traceId, runDuration, evaluationCost,
                Objects.hashCode(additionalProperties));
    }
}
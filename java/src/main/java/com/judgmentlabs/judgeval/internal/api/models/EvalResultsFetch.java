package com.judgmentlabs.judgeval.internal.api.models;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

import com.fasterxml.jackson.annotation.JsonAnyGetter;
import com.fasterxml.jackson.annotation.JsonAnySetter;
import com.fasterxml.jackson.annotation.JsonProperty;

public class EvalResultsFetch {
    @JsonProperty("experiment_run_id")
    private String              experimentRunId;
    @JsonProperty("project_name")
    private String              projectName;

    private Map<String, Object> additionalProperties = new HashMap<>();

    @JsonAnyGetter
    public Map<String, Object> getAdditionalProperties() {
        return additionalProperties;
    }

    @JsonAnySetter
    public void setAdditionalProperty(String name, Object value) {
        additionalProperties.put(name, value);
    }

    public String getExperimentRunId() {
        return experimentRunId;
    }

    public String getProjectName() {
        return projectName;
    }

    public void setExperimentRunId(String experimentRunId) {
        this.experimentRunId = experimentRunId;
    }

    public void setProjectName(String projectName) {
        this.projectName = projectName;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null || getClass() != obj.getClass())
            return false;
        EvalResultsFetch other = (EvalResultsFetch) obj;
        return Objects.equals(experimentRunId, other.experimentRunId) && Objects.equals(projectName, other.projectName)
                && Objects.equals(additionalProperties, other.additionalProperties);
    }

    @Override
    public int hashCode() {
        return Objects.hash(experimentRunId, projectName, Objects.hashCode(additionalProperties));
    }
}
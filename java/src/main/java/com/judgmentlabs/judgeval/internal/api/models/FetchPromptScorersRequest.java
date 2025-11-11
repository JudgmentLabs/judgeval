package com.judgmentlabs.judgeval.internal.api.models;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import com.fasterxml.jackson.annotation.JsonAnyGetter;
import com.fasterxml.jackson.annotation.JsonAnySetter;
import com.fasterxml.jackson.annotation.JsonProperty;

public class FetchPromptScorersRequest {
    @JsonProperty("names")
    private List<String>        names;
    @JsonProperty("is_trace")
    private Boolean             isTrace;

    private Map<String, Object> additionalProperties = new HashMap<>();

    @JsonAnyGetter
    public Map<String, Object> getAdditionalProperties() {
        return additionalProperties;
    }

    @JsonAnySetter
    public void setAdditionalProperty(String name, Object value) {
        additionalProperties.put(name, value);
    }

    public List<String> getNames() {
        return names;
    }

    public Boolean getIsTrace() {
        return isTrace;
    }

    public void setNames(List<String> names) {
        this.names = names;
    }

    public void setIsTrace(Boolean isTrace) {
        this.isTrace = isTrace;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null || getClass() != obj.getClass())
            return false;
        FetchPromptScorersRequest other = (FetchPromptScorersRequest) obj;
        return Objects.equals(names, other.names) && Objects.equals(isTrace, other.isTrace)
                && Objects.equals(additionalProperties, other.additionalProperties);
    }

    @Override
    public int hashCode() {
        return Objects.hash(names, isTrace, Objects.hashCode(additionalProperties));
    }
}
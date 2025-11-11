package com.judgmentlabs.judgeval.internal.api.models;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import com.fasterxml.jackson.annotation.JsonAnyGetter;
import com.fasterxml.jackson.annotation.JsonAnySetter;
import com.fasterxml.jackson.annotation.JsonProperty;

public class OtelTraceSpan {
    @JsonProperty("organization_id")
    private String              organizationId;
    @JsonProperty("project_id")
    private String              projectId;
    @JsonProperty("user_id")
    private String              userId;
    @JsonProperty("timestamp")
    private String              timestamp;
    @JsonProperty("trace_id")
    private String              traceId;
    @JsonProperty("span_id")
    private String              spanId;
    @JsonProperty("parent_span_id")
    private String              parentSpanId;
    @JsonProperty("trace_state")
    private String              traceState;
    @JsonProperty("span_name")
    private String              spanName;
    @JsonProperty("span_kind")
    private String              spanKind;
    @JsonProperty("service_name")
    private String              serviceName;
    @JsonProperty("resource_attributes")
    private Object              resourceAttributes;
    @JsonProperty("span_attributes")
    private Object              spanAttributes;
    @JsonProperty("duration")
    private Integer             duration;
    @JsonProperty("status_code")
    private Integer             statusCode;
    @JsonProperty("status_message")
    private String              statusMessage;
    @JsonProperty("events")
    private List<Object>        events;
    @JsonProperty("links")
    private List<Object>        links;

    private Map<String, Object> additionalProperties = new HashMap<>();

    @JsonAnyGetter
    public Map<String, Object> getAdditionalProperties() {
        return additionalProperties;
    }

    @JsonAnySetter
    public void setAdditionalProperty(String name, Object value) {
        additionalProperties.put(name, value);
    }

    public String getOrganizationId() {
        return organizationId;
    }

    public String getProjectId() {
        return projectId;
    }

    public String getUserId() {
        return userId;
    }

    public String getTimestamp() {
        return timestamp;
    }

    public String getTraceId() {
        return traceId;
    }

    public String getSpanId() {
        return spanId;
    }

    public String getParentSpanId() {
        return parentSpanId;
    }

    public String getTraceState() {
        return traceState;
    }

    public String getSpanName() {
        return spanName;
    }

    public String getSpanKind() {
        return spanKind;
    }

    public String getServiceName() {
        return serviceName;
    }

    public Object getResourceAttributes() {
        return resourceAttributes;
    }

    public Object getSpanAttributes() {
        return spanAttributes;
    }

    public Integer getDuration() {
        return duration;
    }

    public Integer getStatusCode() {
        return statusCode;
    }

    public String getStatusMessage() {
        return statusMessage;
    }

    public List<Object> getEvents() {
        return events;
    }

    public List<Object> getLinks() {
        return links;
    }

    public void setOrganizationId(String organizationId) {
        this.organizationId = organizationId;
    }

    public void setProjectId(String projectId) {
        this.projectId = projectId;
    }

    public void setUserId(String userId) {
        this.userId = userId;
    }

    public void setTimestamp(String timestamp) {
        this.timestamp = timestamp;
    }

    public void setTraceId(String traceId) {
        this.traceId = traceId;
    }

    public void setSpanId(String spanId) {
        this.spanId = spanId;
    }

    public void setParentSpanId(String parentSpanId) {
        this.parentSpanId = parentSpanId;
    }

    public void setTraceState(String traceState) {
        this.traceState = traceState;
    }

    public void setSpanName(String spanName) {
        this.spanName = spanName;
    }

    public void setSpanKind(String spanKind) {
        this.spanKind = spanKind;
    }

    public void setServiceName(String serviceName) {
        this.serviceName = serviceName;
    }

    public void setResourceAttributes(Object resourceAttributes) {
        this.resourceAttributes = resourceAttributes;
    }

    public void setSpanAttributes(Object spanAttributes) {
        this.spanAttributes = spanAttributes;
    }

    public void setDuration(Integer duration) {
        this.duration = duration;
    }

    public void setStatusCode(Integer statusCode) {
        this.statusCode = statusCode;
    }

    public void setStatusMessage(String statusMessage) {
        this.statusMessage = statusMessage;
    }

    public void setEvents(List<Object> events) {
        this.events = events;
    }

    public void setLinks(List<Object> links) {
        this.links = links;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null || getClass() != obj.getClass())
            return false;
        OtelTraceSpan other = (OtelTraceSpan) obj;
        return Objects.equals(organizationId, other.organizationId) && Objects.equals(projectId, other.projectId)
                && Objects.equals(userId, other.userId) && Objects.equals(timestamp, other.timestamp)
                && Objects.equals(traceId, other.traceId) && Objects.equals(spanId, other.spanId)
                && Objects.equals(parentSpanId, other.parentSpanId) && Objects.equals(traceState, other.traceState)
                && Objects.equals(spanName, other.spanName) && Objects.equals(spanKind, other.spanKind)
                && Objects.equals(serviceName, other.serviceName)
                && Objects.equals(resourceAttributes, other.resourceAttributes)
                && Objects.equals(spanAttributes, other.spanAttributes) && Objects.equals(duration, other.duration)
                && Objects.equals(statusCode, other.statusCode) && Objects.equals(statusMessage, other.statusMessage)
                && Objects.equals(events, other.events) && Objects.equals(links, other.links)
                && Objects.equals(additionalProperties, other.additionalProperties);
    }

    @Override
    public int hashCode() {
        return Objects.hash(organizationId, projectId, userId, timestamp, traceId, spanId, parentSpanId, traceState,
                spanName, spanKind, serviceName, resourceAttributes, spanAttributes, duration, statusCode,
                statusMessage, events, links, Objects.hashCode(additionalProperties));
    }
}
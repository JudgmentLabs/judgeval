package com.judgmentlabs.judgeval.v1.scorers;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.judgmentlabs.judgeval.internal.api.models.ScorerConfig;
import com.judgmentlabs.judgeval.v1.data.APIScorerType;

/**
 * Base class for API-based scorers that evaluate using the Judgment backend.
 */
public class APIScorer extends com.judgmentlabs.judgeval.internal.api.models.BaseScorer implements BaseScorer {
    private APIScorerType scoreType;

    @JsonIgnore
    private List<String>  requiredParams;

    public APIScorer(APIScorerType scoreType) {
        super();
        this.scoreType = scoreType;
        setName(scoreType.toString());
        setScoreType(scoreType.toString());
        this.requiredParams = new java.util.ArrayList<>();
        if (Boolean.TRUE.equals(getStrictMode())) {
            setThreshold(1.0);
        }
    }

    public void setThreshold(double threshold) {
        if (threshold < 0 || threshold > 1) {
            throw new IllegalArgumentException("Threshold must be between 0 and 1, got: " + threshold);
        }
        super.setThreshold(threshold);
    }

    @JsonProperty("score_type")
    public String getScoreType() {
        return scoreType.toString();
    }

    public List<String> getRequiredParams() {
        return requiredParams;
    }

    public void setRequiredParams(List<String> requiredParams) {
        this.requiredParams = requiredParams;
    }

    @Override
    public Double getThreshold() {
        return Optional.ofNullable(super.getThreshold())
                .orElse(0.5);
    }

    @Override
    public String getName() {
        return Optional.ofNullable(super.getName())
                .map(Object::toString)
                .orElse(null);
    }

    @Override
    public Boolean getStrictMode() {
        return Optional.ofNullable(super.getStrictMode())
                .orElse(false);
    }

    @Override
    @JsonIgnore
    public ScorerConfig getScorerConfig() {
        ScorerConfig cfg = new ScorerConfig();
        cfg.setScoreType(getScoreType());
        cfg.setThreshold(getThreshold());
        cfg.setName(getName());
        cfg.setStrictMode(getStrictMode());
        cfg.setRequiredParams(getRequiredParams());
        Map<String, Object> kwargs = new HashMap<>();
        if (getAdditionalProperties() != null)
            kwargs.putAll(getAdditionalProperties());
        cfg.setKwargs(kwargs);
        return cfg;
    }

    /**
     * Creates a new builder for an APIScorer subclass.
     *
     * @param <T>
     *            the scorer type
     * @param scorerClass
     *            the scorer class
     * @return a new builder instance
     */
    public static <T extends APIScorer> Builder<T> builder(Class<T> scorerClass) {
        return new Builder<>(scorerClass);
    }

    /**
     * Builder for configuring and creating APIScorer instances.
     *
     * @param <T>
     *            the scorer type
     */
    public static final class Builder<T extends APIScorer> {
        private final T scorer;

        private Builder(Class<T> scorerClass) {
            try {
                this.scorer = scorerClass.getDeclaredConstructor()
                        .newInstance();
            } catch (Exception e) {
                throw new RuntimeException("Failed to create scorer instance", e);
            }
        }

        public Builder<T> threshold(double threshold) {
            scorer.setThreshold(threshold);
            return this;
        }

        public Builder<T> name(String name) {
            scorer.setName(name);
            return this;
        }

        public Builder<T> strictMode(boolean strictMode) {
            scorer.setStrictMode(strictMode);
            return this;
        }

        public Builder<T> requiredParams(List<String> requiredParams) {
            scorer.setRequiredParams(requiredParams);
            return this;
        }

        public Builder<T> model(String model) {
            scorer.setModel(model);
            return this;
        }

        public Builder<T> additionalProperty(String key, Object value) {
            scorer.setAdditionalProperty(key, value);
            return this;
        }

        public T build() {
            return scorer;
        }
    }
}

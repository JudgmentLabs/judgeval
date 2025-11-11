package com.judgmentlabs.judgeval.v1.scorers.prompt_scorer;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

import com.judgmentlabs.judgeval.internal.api.models.ScorerConfig;
import com.judgmentlabs.judgeval.v1.data.APIScorerType;
import com.judgmentlabs.judgeval.v1.scorers.APIScorer;

/**
 * Scorer that evaluates traces using Judgment-hosted prompt scorers.
 * 
 * Prompt scorers are hosted on Judgment Servers and can be configured using the
 * Scorer Playground.
 * 
 * @see <a href=
 *      "https://docs.judgmentlabs.ai/documentation/evaluation/prompt-scorers">Judgment
 *      Docs: Prompt Scorers</a>
 */
public final class PromptScorer extends APIScorer {
    private final String              prompt;
    private final Map<String, Double> options;
    private final boolean             isTrace;

    private PromptScorer(Builder builder) {
        super(builder.isTrace ? APIScorerType.TRACE_PROMPT_SCORER : APIScorerType.PROMPT_SCORER);
        this.prompt = Objects.requireNonNull(builder.prompt, "prompt required");
        this.options = builder.options;
        this.isTrace = builder.isTrace;
        setName(Objects.requireNonNull(builder.name, "name required"));
        setThreshold(builder.threshold);
    }

    public String getPrompt() {
        return prompt;
    }

    public Map<String, Double> getOptions() {
        return options != null ? new HashMap<>(options) : null;
    }

    public String getScorerName() {
        return getName();
    }

    @Override
    public ScorerConfig getScorerConfig() {
        ScorerConfig cfg = new ScorerConfig();
        cfg.setScoreType(getScoreType());
        cfg.setThreshold(getThreshold());
        cfg.setName(getName());
        cfg.setStrictMode(getStrictMode());
        cfg.setRequiredParams(getRequiredParams());

        Map<String, Object> kwargs = new HashMap<>();
        kwargs.put("prompt", prompt);
        if (options != null) {
            kwargs.put("options", options);
        }
        if (getAdditionalProperties() != null) {
            kwargs.putAll(getAdditionalProperties());
        }
        cfg.setKwargs(kwargs);
        return cfg;
    }

    /**
     * Creates a new builder for configuring a PromptScorer.
     *
     * @return a new builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for configuring and creating PromptScorer instances.
     */
    public static final class Builder {
        private String              name;
        private String              prompt;
        private double              threshold = 0.5;
        private Map<String, Double> options;
        private boolean             isTrace;

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder prompt(String prompt) {
            this.prompt = prompt;
            return this;
        }

        public Builder threshold(double threshold) {
            this.threshold = threshold;
            return this;
        }

        public Builder options(Map<String, Double> options) {
            this.options = options;
            return this;
        }

        Builder isTrace(boolean isTrace) {
            this.isTrace = isTrace;
            return this;
        }

        public PromptScorer build() {
            return new PromptScorer(this);
        }
    }

    @Override
    public String toString() {
        return "PromptScorer(name=" + getName() + ", prompt=" + prompt + ", threshold=" + getThreshold()
                + ", options=" + options + ", isTrace=" + isTrace + ")";
    }
}

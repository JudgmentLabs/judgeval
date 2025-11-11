package com.judgmentlabs.judgeval.scorers.api_scorers.prompt_scorer;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

import com.judgmentlabs.judgeval.Env;
import com.judgmentlabs.judgeval.data.APIScorerType;
import com.judgmentlabs.judgeval.exceptions.JudgmentAPIError;
import com.judgmentlabs.judgeval.internal.api.models.ScorerConfig;

/**
 * @deprecated Use {@link com.judgmentlabs.judgeval.v1.Judgeval} instead.
 * 
 *             <p>
 *             Migration example:
 * 
 *             <pre>{@code
 * // Old way:
 * TracePromptScorer scorer = TracePromptScorer.get("my-scorer");
 * 
 * // New way:
 * Judgeval client = Judgeval.builder().build();
 * PromptScorer scorer = client.scorers().tracePromptScorer().get("my-scorer");
 * }</pre>
 */
@Deprecated
public class TracePromptScorer extends BasePromptScorer {

    public TracePromptScorer(String name, String prompt, double threshold, Map<String, Double> options) {
        super(APIScorerType.TRACE_PROMPT_SCORER, name, prompt, threshold, options, Env.JUDGMENT_API_KEY,
                Env.JUDGMENT_ORG_ID);
    }

    public TracePromptScorer(String name, String prompt, double threshold, Map<String, Double> options,
            String judgmentApiKey, String organizationId) {
        super(APIScorerType.TRACE_PROMPT_SCORER, name, prompt, threshold, options, judgmentApiKey, organizationId);
    }

    public static TracePromptScorer get(String name) {
        return get(name, Env.JUDGMENT_API_KEY, Env.JUDGMENT_ORG_ID);
    }

    public static TracePromptScorer get(String name, String judgmentApiKey, String organizationId) {
        com.judgmentlabs.judgeval.internal.api.models.PromptScorer scorerConfig = fetchPromptScorer(name,
                judgmentApiKey, organizationId);

        if (!Boolean.TRUE.equals(scorerConfig.getIsTrace())) {
            throw new JudgmentAPIError(400, "Scorer with name " + name + " is not a TracePromptScorer");
        }

        Map<String, Double> options = null;
        if (scorerConfig.getOptions() != null) {
            if (scorerConfig.getOptions() instanceof Map) {
                @SuppressWarnings("unchecked")
                Map<String, Object> rawOptions = (Map<String, Object>) scorerConfig.getOptions();
                options = new HashMap<>();
                for (Map.Entry<String, Object> entry : rawOptions.entrySet()) {
                    if (entry.getValue() instanceof Number) {
                        options.put(entry.getKey(), ((Number) entry.getValue()).doubleValue());
                    }
                }
            }
        }

        return new TracePromptScorer(name, scorerConfig.getPrompt(),
                Optional.ofNullable(scorerConfig.getThreshold())
                        .orElse(0.5),
                options, judgmentApiKey, organizationId);
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

    @Override
    protected boolean isTrace() {
        return true;
    }
}

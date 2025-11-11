package com.judgmentlabs.judgeval.scorers.api_scorers.prompt_scorer;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

import com.judgmentlabs.judgeval.Env;
import com.judgmentlabs.judgeval.data.APIScorerType;
import com.judgmentlabs.judgeval.exceptions.JudgmentAPIError;
import com.judgmentlabs.judgeval.internal.api.JudgmentSyncClient;
import com.judgmentlabs.judgeval.internal.api.models.FetchPromptScorersRequest;
import com.judgmentlabs.judgeval.internal.api.models.FetchPromptScorersResponse;
import com.judgmentlabs.judgeval.internal.api.models.ScorerExistsRequest;
import com.judgmentlabs.judgeval.internal.api.models.ScorerExistsResponse;
import com.judgmentlabs.judgeval.scorers.APIScorer;

public abstract class BasePromptScorer extends APIScorer {
    private static final Map<CacheKey, com.judgmentlabs.judgeval.internal.api.models.PromptScorer> cache = new ConcurrentHashMap<>();

    protected String                                                                               prompt;
    protected Map<String, Double>                                                                  options;
    protected String                                                                               judgmentApiKey;
    protected String                                                                               organizationId;

    protected BasePromptScorer(APIScorerType scoreType, String name, String prompt, double threshold,
            Map<String, Double> options, String judgmentApiKey, String organizationId) {
        super(scoreType);
        this.prompt = prompt;
        this.options = options;
        this.judgmentApiKey = judgmentApiKey;
        this.organizationId = organizationId;
        setName(name);
        super.setThreshold(threshold);
    }

    public static boolean scorerExists(String name, String judgmentApiKey, String organizationId) {
        try {
            JudgmentSyncClient client = new JudgmentSyncClient(Env.JUDGMENT_API_URL, judgmentApiKey, organizationId);
            ScorerExistsRequest request = new ScorerExistsRequest();
            request.setName(name);
            ScorerExistsResponse response = client.scorerExists(request);
            return Boolean.TRUE.equals(response.getExists());
        } catch (Exception e) {
            throw new JudgmentAPIError(500, "Failed to check if scorer exists: " + e.getMessage());
        }
    }

    public static com.judgmentlabs.judgeval.internal.api.models.PromptScorer fetchPromptScorer(String name,
            String judgmentApiKey, String organizationId) {
        CacheKey key = new CacheKey(name, judgmentApiKey, organizationId);
        com.judgmentlabs.judgeval.internal.api.models.PromptScorer cached = cache.get(key);
        if (cached != null) {
            return cached;
        }

        try {
            JudgmentSyncClient client = new JudgmentSyncClient(Env.JUDGMENT_API_URL, judgmentApiKey, organizationId);
            FetchPromptScorersRequest request = new FetchPromptScorersRequest();
            request.setNames(java.util.Collections.singletonList(name));

            FetchPromptScorersResponse response = client.fetchScorers(request);

            com.judgmentlabs.judgeval.internal.api.models.PromptScorer scorer = Optional.ofNullable(response)
                    .map(FetchPromptScorersResponse::getScorers)
                    .filter(scorers -> scorers != null && !scorers.isEmpty())
                    .map(scorers -> scorers.get(0))
                    .orElseThrow(
                            () -> new JudgmentAPIError(404, "Failed to fetch prompt scorer '" + name + "': not found"));

            cache.put(key, scorer);
            return scorer;
        } catch (JudgmentAPIError e) {
            throw e;
        } catch (Exception e) {
            throw new JudgmentAPIError(500, "Failed to fetch prompt scorer '" + name + "': " + e.getMessage());
        }
    }

    public Double getThreshold() {
        return super.getThreshold();
    }

    public String getPrompt() {
        return prompt;
    }

    public Map<String, Double> getOptions() {
        return Optional.ofNullable(options)
                .map(HashMap::new)
                .orElse(null);
    }

    public String getScorerName() {
        return getName();
    }

    public Map<String, Object> getConfig() {
        Map<String, Object> config = new HashMap<>();
        config.put("name", getName());
        config.put("prompt", prompt);
        config.put("threshold", getThreshold());
        config.put("options", options);
        return config;
    }

    protected abstract boolean isTrace();

    private static final class CacheKey {
        private final String name;
        private final String apiKey;
        private final String organizationId;

        CacheKey(String name, String apiKey, String organizationId) {
            this.name = name;
            this.apiKey = apiKey;
            this.organizationId = organizationId;
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj)
                return true;
            if (obj == null || getClass() != obj.getClass())
                return false;
            CacheKey that = (CacheKey) obj;
            return Objects.equals(name, that.name) && Objects.equals(apiKey, that.apiKey)
                    && Objects.equals(organizationId, that.organizationId);
        }

        @Override
        public int hashCode() {
            return Objects.hash(name, apiKey, organizationId);
        }
    }

    @Override
    public String toString() {
        return "PromptScorer(name=" + getName() + ", prompt=" + prompt + ", threshold=" + getThreshold()
                + ", options=" + options + ")";
    }
}

package com.judgmentlabs.judgeval.scorers.prompt_scorer;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

import com.judgmentlabs.judgeval.internal.api.JudgmentSyncClient;
import com.judgmentlabs.judgeval.internal.api.models.FetchPromptScorersRequest;
import com.judgmentlabs.judgeval.internal.api.models.FetchPromptScorersResponse;
import com.judgmentlabs.judgeval.utils.Logger;

/**
 * Factory for retrieving prompt-based scorers.
 */
public final class PromptScorerFactory {
    private final JudgmentSyncClient                                                               client;
    private final boolean                                                                          isTrace;
    private static final Map<CacheKey, com.judgmentlabs.judgeval.internal.api.models.PromptScorer> cache = new ConcurrentHashMap<>();

    public PromptScorerFactory(JudgmentSyncClient client, boolean isTrace) {
        this.client = client;
        this.isTrace = isTrace;
    }

    /**
     * Retrieves a prompt scorer by name from the Judgment API.
     * Results are cached to avoid repeated API calls.
     *
     * @param name
     *            the scorer name
     * @return the configured prompt scorer or null if not found or retrieval fails
     */
    public PromptScorer get(String name) {
        CacheKey key = new CacheKey(name, client.getApiKey(), client.getOrganizationId());
        com.judgmentlabs.judgeval.internal.api.models.PromptScorer cached = cache.get(key);
        if (cached != null) {
            return createFromModel(cached, name);
        }

        try {
            FetchPromptScorersRequest request = new FetchPromptScorersRequest();
            request.setNames(java.util.Collections.singletonList(name));

            FetchPromptScorersResponse response = client.fetchScorers(request);

            com.judgmentlabs.judgeval.internal.api.models.PromptScorer scorer = Optional.ofNullable(response)
                    .map(FetchPromptScorersResponse::getScorers)
                    .filter(scorers -> scorers != null && !scorers.isEmpty())
                    .map(scorers -> scorers.get(0))
                    .orElseGet(
                            () -> {
                                Logger.error("Failed to fetch prompt scorer '" + name + "': not found");
                                return null;
                            });

            if (scorer == null) {
                return null;
            }

            if (Boolean.TRUE.equals(scorer.getIsTrace()) != isTrace) {
                Logger.error("Scorer with name " + name + " is a "
                        + (Boolean.TRUE.equals(scorer.getIsTrace()) ? "TracePromptScorer" : "PromptScorer") + ", not a "
                        + (isTrace ? "TracePromptScorer" : "PromptScorer"));
                return null;
            }

            cache.put(key, scorer);
            return createFromModel(scorer, name);
        } catch (Exception e) {
            Logger.error("Failed to fetch prompt scorer '" + name + "': " + e.getMessage());
            return null;
        }
    }

    private PromptScorer createFromModel(com.judgmentlabs.judgeval.internal.api.models.PromptScorer model,
            String name) {
        Map<String, Double> options = null;
        if (model.getOptions() != null) {
            if (model.getOptions() instanceof Map) {
                @SuppressWarnings("unchecked")
                Map<String, Object> rawOptions = (Map<String, Object>) model.getOptions();
                options = new HashMap<>();
                for (Map.Entry<String, Object> entry : rawOptions.entrySet()) {
                    if (entry.getValue() instanceof Number) {
                        options.put(entry.getKey(), ((Number) entry.getValue()).doubleValue());
                    }
                }
            }
        }

        return PromptScorer.builder()
                .name(name)
                .prompt(model.getPrompt())
                .threshold(Optional.ofNullable(model.getThreshold()).orElse(0.5))
                .options(options)
                .isTrace(isTrace)
                .build();
    }

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
}

package com.judgmentlabs.judgeval.scorers;

import com.judgmentlabs.judgeval.internal.api.models.ScorerConfig;

/**
 * Minimal interface for scorers used by BaseTracer. Only requires the essential
 * methods needed for evaluation.
 * 
 * @deprecated Use {@link com.judgmentlabs.judgeval.v1.scorers.BaseScorer}
 *             instead.
 */
@Deprecated
public interface BaseScorer {
    /**
     * Gets the name of the scorer.
     *
     * @return the scorer name
     */
    String getName();

    /**
     * Gets the scorer configuration for evaluation runs.
     *
     * @return the scorer configuration
     */
    ScorerConfig getScorerConfig();
}

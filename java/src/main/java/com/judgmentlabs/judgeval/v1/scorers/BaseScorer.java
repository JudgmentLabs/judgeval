package com.judgmentlabs.judgeval.v1.scorers;

import com.judgmentlabs.judgeval.internal.api.models.ScorerConfig;

/**
 * Base interface for all scorers in the Judgment evaluation system.
 */
public interface BaseScorer {
    /**
     * Returns the name of this scorer.
     *
     * @return the scorer name
     */
    String getName();

    /**
     * Returns the configuration for this scorer.
     *
     * @return the scorer configuration
     */
    ScorerConfig getScorerConfig();
}

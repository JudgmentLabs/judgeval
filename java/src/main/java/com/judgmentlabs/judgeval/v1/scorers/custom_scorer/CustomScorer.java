package com.judgmentlabs.judgeval.v1.scorers.custom_scorer;

import com.judgmentlabs.judgeval.internal.api.models.ScorerConfig;
import com.judgmentlabs.judgeval.v1.data.APIScorerType;
import com.judgmentlabs.judgeval.v1.scorers.APIScorer;

/**
 * Scorer that uses custom user-defined evaluation logic. Scorers are hosted on
 * Judgment Servers
 * and can be uploaded using the judgeval CLI.
 * 
 * @see <a href="https://docs.judgment.ai/judgeval/cli/upload-scorers">Judgment
 *      Docs: Upload Scorers</a>
 */
public final class CustomScorer extends APIScorer {
    private CustomScorer(Builder builder) {
        super(APIScorerType.CUSTOM);
        setName(builder.name);
        setClassName(builder.className);
        // Java SDK only supports server-hosted scorers
        setServerHosted(true);
    }

    @Override
    public ScorerConfig getScorerConfig() {
        throw new UnsupportedOperationException("CustomScorer does not use ScorerConfig");
    }

    /**
     * Creates a new builder for configuring a CustomScorer.
     *
     * @return a new builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for configuring and creating CustomScorer instances.
     */
    public static final class Builder {
        private String name;
        private String className;

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder className(String className) {
            this.className = className;
            return this;
        }

        public CustomScorer build() {
            return new CustomScorer(this);
        }
    }
}

package com.judgmentlabs.judgeval.data;

import java.util.List;

/**
 * @deprecated Use {@link com.judgmentlabs.judgeval.v1.data.ScoringResult}
 *             instead.
 */
@Deprecated
public class ScoringResult extends com.judgmentlabs.judgeval.internal.api.models.ScoringResult {

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private final ScoringResult result;

        private Builder() {
            this.result = new ScoringResult();
        }

        public Builder success(Boolean success) {
            result.setSuccess(success);
            return this;
        }

        public Builder scorersData(List<ScorerData> scorersData) {
            @SuppressWarnings("unchecked")
            List<com.judgmentlabs.judgeval.internal.api.models.ScorerData> internalList = (List<com.judgmentlabs.judgeval.internal.api.models.ScorerData>) (List<?>) scorersData;
            result.setScorersData(internalList);
            return this;
        }

        public Builder scorerData(ScorerData scorerData) {
            if (result.getScorersData() == null) {
                result.setScorersData(new java.util.ArrayList<>());
            }
            result.getScorersData()
                    .add(scorerData);
            return this;
        }

        public Builder dataObject(Example dataObject) {
            // Store Example in additional properties since setDataObject
            // expects TraceSpan
            // This indicates a potential API design issue - ScoringResult may
            // be
            // trace-specific
            if (dataObject != null) {
                result.setAdditionalProperty("example", dataObject);
            }
            return this;
        }

        public ScoringResult build() {
            return result;
        }
    }
}

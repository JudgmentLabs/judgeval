package com.judgmentlabs.judgeval.scorers.api_scorers;

import java.util.Arrays;

import com.judgmentlabs.judgeval.data.APIScorerType;
import com.judgmentlabs.judgeval.scorers.APIScorer;

/**
 * @deprecated Use {@link com.judgmentlabs.judgeval.v1.Judgeval} instead.
 * 
 *             <p>
 *             Migration example:
 * 
 *             <pre>{@code
 * // Old way:
 * AnswerRelevancyScorer scorer = AnswerRelevancyScorer.create();
 * 
 * // New way:
 * Judgeval client = Judgeval.builder().build();
 * AnswerRelevancyScorer scorer = client.scorers().builtIn().answerRelevancy().build();
 * }</pre>
 */
@Deprecated
public class AnswerRelevancyScorer extends APIScorer {
    public AnswerRelevancyScorer() {
        super(APIScorerType.ANSWER_RELEVANCY);
        setRequiredParams(Arrays.asList("input", "actual_output"));
    }

    public static APIScorer.Builder<AnswerRelevancyScorer> builder() {
        return APIScorer.builder(AnswerRelevancyScorer.class);
    }

    public static AnswerRelevancyScorer create() {
        return new AnswerRelevancyScorer();
    }

    public static AnswerRelevancyScorer create(double threshold) {
        return builder().threshold(threshold)
                .build();
    }
}

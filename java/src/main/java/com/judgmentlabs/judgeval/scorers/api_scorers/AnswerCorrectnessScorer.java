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
 * AnswerCorrectnessScorer scorer = AnswerCorrectnessScorer.create();
 * 
 * // New way:
 * Judgeval client = Judgeval.builder().build();
 * AnswerCorrectnessScorer scorer = client.scorers().builtIn().answerCorrectness().build();
 * }</pre>
 */
@Deprecated
public class AnswerCorrectnessScorer extends APIScorer {
    public AnswerCorrectnessScorer() {
        super(APIScorerType.ANSWER_CORRECTNESS);
        setRequiredParams(Arrays.asList("input", "actual_output", "expected_output"));
    }

    public static APIScorer.Builder<AnswerCorrectnessScorer> builder() {
        return APIScorer.builder(AnswerCorrectnessScorer.class);
    }

    public static AnswerCorrectnessScorer create() {
        return new AnswerCorrectnessScorer();
    }

    public static AnswerCorrectnessScorer create(double threshold) {
        return builder().threshold(threshold)
                .build();
    }
}

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
 * InstructionAdherenceScorer scorer = InstructionAdherenceScorer.create();
 * 
 * // New way:
 * Judgeval client = Judgeval.builder().build();
 * InstructionAdherenceScorer scorer = client.scorers().builtIn().instructionAdherence().build();
 * }</pre>
 */
@Deprecated
public class InstructionAdherenceScorer extends APIScorer {
    public InstructionAdherenceScorer() {
        super(APIScorerType.INSTRUCTION_ADHERENCE);
        setRequiredParams(Arrays.asList("input", "actual_output"));
        setName("Instruction Adherence");
    }

    public static APIScorer.Builder<InstructionAdherenceScorer> builder() {
        return APIScorer.builder(InstructionAdherenceScorer.class);
    }

    public static InstructionAdherenceScorer create() {
        return new InstructionAdherenceScorer();
    }

    public static InstructionAdherenceScorer create(double threshold) {
        return builder().threshold(threshold)
                .build();
    }
}

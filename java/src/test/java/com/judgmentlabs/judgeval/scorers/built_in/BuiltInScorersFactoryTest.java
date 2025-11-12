package com.judgmentlabs.judgeval.scorers.built_in;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class BuiltInScorersFactoryTest {
    private BuiltInScorersFactory factory;

    @BeforeEach
    void setUp() {
        factory = new BuiltInScorersFactory();
    }

    @Test
    void answerCorrectness_returnsBuilder() {
        AnswerCorrectnessScorer.Builder builder = factory.answerCorrectness();
        assertNotNull(builder);
    }

    @Test
    void answerRelevancy_returnsBuilder() {
        AnswerRelevancyScorer.Builder builder = factory.answerRelevancy();
        assertNotNull(builder);
    }

    @Test
    void faithfulness_returnsBuilder() {
        FaithfulnessScorer.Builder builder = factory.faithfulness();
        assertNotNull(builder);
    }
}

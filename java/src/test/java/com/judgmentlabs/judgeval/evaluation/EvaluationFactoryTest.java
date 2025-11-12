package com.judgmentlabs.judgeval.evaluation;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import com.judgmentlabs.judgeval.internal.api.JudgmentSyncClient;

class EvaluationFactoryTest {
    private static final String TEST_API_URL = "https://api.test.com";
    private static final String TEST_API_KEY = "test-key";
    private static final String TEST_ORG_ID  = "test-org";

    private EvaluationFactory   factory;

    @BeforeEach
    void setUp() {
        JudgmentSyncClient client = new JudgmentSyncClient(TEST_API_URL, TEST_API_KEY, TEST_ORG_ID);
        factory = new EvaluationFactory(client);
    }

    @Test
    void create_returnsConfiguredBuilder() {
        Evaluation.Builder builder = factory.create();
        assertNotNull(builder);
    }

    @Test
    void create_builderBuildsEvaluation() {
        Evaluation.Builder builder = factory.create();
        Evaluation evaluation = builder.build();
        assertNotNull(evaluation);
    }
}

package com.judgmentlabs.judgeval;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

import com.judgmentlabs.judgeval.evaluation.EvaluationFactory;
import com.judgmentlabs.judgeval.scorers.ScorersFactory;
import com.judgmentlabs.judgeval.tracer.TracerFactory;

class JudgevalTest {
    private static final String TEST_API_URL = "https://api.test.com";
    private static final String TEST_API_KEY = "test-key";
    private static final String TEST_ORG_ID  = "test-org";

    @Test
    void builder_withAllParameters_buildsSuccessfully() {
        Judgeval client = Judgeval.builder()
                .apiKey(TEST_API_KEY)
                .organizationId(TEST_ORG_ID)
                .apiUrl(TEST_API_URL)
                .build();

        assertNotNull(client);
    }

    @Test
    void builder_withNullApiKey_throwsException() {
        assertThrows(NullPointerException.class, () -> {
            Judgeval.builder()
                    .apiKey(null)
                    .organizationId(TEST_ORG_ID)
                    .build();
        });
    }

    @Test
    void builder_withNullOrganizationId_throwsException() {
        assertThrows(NullPointerException.class, () -> {
            Judgeval.builder()
                    .apiKey(TEST_API_KEY)
                    .organizationId(null)
                    .build();
        });
    }

    @Test
    void tracer_returnsTracerFactory() {
        Judgeval client = Judgeval.builder()
                .apiKey(TEST_API_KEY)
                .organizationId(TEST_ORG_ID)
                .build();

        TracerFactory factory = client.tracer();
        assertNotNull(factory);
    }

    @Test
    void scorers_returnsScorersFactory() {
        Judgeval client = Judgeval.builder()
                .apiKey(TEST_API_KEY)
                .organizationId(TEST_ORG_ID)
                .build();

        ScorersFactory factory = client.scorers();
        assertNotNull(factory);
    }

    @Test
    void evaluation_returnsEvaluationFactory() {
        Judgeval client = Judgeval.builder()
                .apiKey(TEST_API_KEY)
                .organizationId(TEST_ORG_ID)
                .build();

        EvaluationFactory factory = client.evaluation();
        assertNotNull(factory);
    }
}

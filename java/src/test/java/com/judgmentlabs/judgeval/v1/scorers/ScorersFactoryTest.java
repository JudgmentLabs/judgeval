package com.judgmentlabs.judgeval.v1.scorers;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import com.judgmentlabs.judgeval.internal.api.JudgmentSyncClient;
import com.judgmentlabs.judgeval.v1.scorers.built_in.BuiltInScorersFactory;
import com.judgmentlabs.judgeval.v1.scorers.custom_scorer.CustomScorerFactory;
import com.judgmentlabs.judgeval.v1.scorers.prompt_scorer.PromptScorerFactory;

class ScorersFactoryTest {
    private static final String TEST_API_URL = "https://api.test.com";
    private static final String TEST_API_KEY = "test-key";
    private static final String TEST_ORG_ID  = "test-org";

    private ScorersFactory      factory;

    @BeforeEach
    void setUp() {
        JudgmentSyncClient client = new JudgmentSyncClient(TEST_API_URL, TEST_API_KEY, TEST_ORG_ID);
        factory = new ScorersFactory(client);
    }

    @Test
    void promptScorer_returnsFactory() {
        PromptScorerFactory promptScorerFactory = factory.promptScorer();
        assertNotNull(promptScorerFactory);
    }

    @Test
    void tracePromptScorer_returnsFactory() {
        PromptScorerFactory tracePromptScorerFactory = factory.tracePromptScorer();
        assertNotNull(tracePromptScorerFactory);
    }

    @Test
    void customScorer_returnsFactory() {
        CustomScorerFactory customScorerFactory = factory.customScorer();
        assertNotNull(customScorerFactory);
    }

    @Test
    void builtIn_returnsFactory() {
        BuiltInScorersFactory builtInFactory = factory.builtIn();
        assertNotNull(builtInFactory);
    }
}

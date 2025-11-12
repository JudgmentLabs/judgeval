package com.judgmentlabs.judgeval.tracer;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import com.judgmentlabs.judgeval.internal.api.JudgmentSyncClient;

class TracerFactoryTest {
    private static final String TEST_API_URL = "https://api.test.com";
    private static final String TEST_API_KEY = "test-key";
    private static final String TEST_ORG_ID  = "test-org";

    private TracerFactory       factory;

    @BeforeEach
    void setUp() {
        JudgmentSyncClient client = new JudgmentSyncClient(TEST_API_URL, TEST_API_KEY, TEST_ORG_ID);
        factory = new TracerFactory(client);
    }

    @Test
    void create_returnsConfiguredBuilder() {
        Tracer.Builder builder = factory.create();
        assertNotNull(builder);
    }
}

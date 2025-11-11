package com.judgmentlabs.judgeval.v1.tracer;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.lenient;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import com.judgmentlabs.judgeval.internal.api.JudgmentSyncClient;
import com.judgmentlabs.judgeval.internal.api.models.ResolveProjectNameRequest;
import com.judgmentlabs.judgeval.internal.api.models.ResolveProjectNameResponse;

@ExtendWith(MockitoExtension.class)
class TracerTest {
    private static final String TEST_PROJECT_NAME = "test-project";
    private static final String TEST_PROJECT_ID   = "test-project-id-123";

    @Mock
    private JudgmentSyncClient  mockClient;

    @BeforeEach
    void setUp() throws Exception {
        ResolveProjectNameResponse response = new ResolveProjectNameResponse();
        response.setProjectId(TEST_PROJECT_ID);

        lenient().when(mockClient.projectsResolve(any(ResolveProjectNameRequest.class)))
                .thenReturn(response);
    }

    @Test
    void builder_returnsBuilder() {
        Tracer.Builder builder = Tracer.builder();
        assertNotNull(builder);
    }

    @Test
    void builder_withNullProjectName_throwsException() {
        assertThrows(NullPointerException.class, () -> {
            Tracer.builder()
                    .client(mockClient)
                    .projectName(null)
                    .build();
        });
    }

    @Test
    void builder_withNullClient_throwsException() {
        assertThrows(NullPointerException.class, () -> {
            Tracer.builder()
                    .projectName(TEST_PROJECT_NAME)
                    .client(null)
                    .build();
        });
    }
}

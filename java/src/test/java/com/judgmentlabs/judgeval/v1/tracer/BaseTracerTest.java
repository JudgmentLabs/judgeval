package com.judgmentlabs.judgeval.v1.tracer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.lenient;
import static org.mockito.Mockito.when;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import com.judgmentlabs.judgeval.internal.api.JudgmentSyncClient;
import com.judgmentlabs.judgeval.internal.api.models.ResolveProjectNameRequest;
import com.judgmentlabs.judgeval.internal.api.models.ResolveProjectNameResponse;
import com.judgmentlabs.judgeval.v1.tracer.exporters.JudgmentSpanExporter;
import com.judgmentlabs.judgeval.v1.tracer.exporters.NoOpSpanExporter;

import io.opentelemetry.sdk.trace.export.SpanExporter;

@ExtendWith(MockitoExtension.class)
class BaseTracerTest {
    private static final String TEST_PROJECT_NAME = "test-project";
    private static final String TEST_PROJECT_ID   = "test-project-id-123";
    @Mock
    private JudgmentSyncClient  mockClient;

    @Mock
    private ISerializer         mockSerializer;

    private TestableBaseTracer  tracer;

    @BeforeEach
    void setUp() throws Exception {
        ResolveProjectNameResponse response = new ResolveProjectNameResponse();
        response.setProjectId(TEST_PROJECT_ID);

        lenient().when(mockClient.projectsResolve(any(ResolveProjectNameRequest.class)))
                .thenReturn(response);
        lenient().when(mockClient.getApiUrl()).thenReturn("https://api.example.com");
        lenient().when(mockClient.getApiKey()).thenReturn("test-api-key");
        lenient().when(mockClient.getOrganizationId()).thenReturn("test-org-id");

        lenient().when(mockSerializer.serialize(any())).thenReturn("serialized");
        lenient().when(mockSerializer.serialize(any(), any())).thenReturn("serialized");

        tracer = new TestableBaseTracer(
                TEST_PROJECT_NAME,
                true,
                mockClient,
                mockSerializer);
    }

    @Test
    void constructor_withValidParameters_resolvesProject() {
        assertNotNull(tracer);
        assertEquals(TEST_PROJECT_NAME, tracer.getProjectName());
        assertTrue(tracer.isEnableEvaluation());
        assertTrue(tracer.getProjectId().isPresent());
        assertEquals(TEST_PROJECT_ID, tracer.getProjectId().get());
    }

    @Test
    void constructor_withNullProjectName_throwsException() {
        assertThrows(NullPointerException.class, () -> {
            new TestableBaseTracer(
                    null,
                    true,
                    mockClient,
                    mockSerializer);
        });
    }

    @Test
    void constructor_withNullClient_throwsException() {
        assertThrows(NullPointerException.class, () -> {
            new TestableBaseTracer(
                    TEST_PROJECT_NAME,
                    true,
                    null,
                    mockSerializer);
        });
    }

    @Test
    void constructor_withNullSerializer_throwsException() {
        assertThrows(NullPointerException.class, () -> {
            new TestableBaseTracer(
                    TEST_PROJECT_NAME,
                    true,
                    mockClient,
                    null);
        });
    }

    @Test
    void constructor_withFailedProjectResolution_hasEmptyProjectId() throws Exception {
        when(mockClient.projectsResolve(any(ResolveProjectNameRequest.class)))
                .thenThrow(new RuntimeException("Project not found"));

        TestableBaseTracer failedTracer = new TestableBaseTracer(
                TEST_PROJECT_NAME,
                true,
                mockClient,
                mockSerializer);

        assertFalse(failedTracer.getProjectId().isPresent());
    }

    @Test
    void getSpanExporter_withValidProjectId_returnsJudgmentSpanExporter() {
        SpanExporter exporter = tracer.getSpanExporter();
        assertNotNull(exporter);
        assertTrue(exporter instanceof JudgmentSpanExporter);
    }

    @Test
    void getSpanExporter_withoutProjectId_returnsNoOpSpanExporter() throws Exception {
        when(mockClient.projectsResolve(any(ResolveProjectNameRequest.class)))
                .thenReturn(null);

        TestableBaseTracer failedTracer = new TestableBaseTracer(
                TEST_PROJECT_NAME,
                true,
                mockClient,
                mockSerializer);

        SpanExporter exporter = failedTracer.getSpanExporter();
        assertNotNull(exporter);
        assertTrue(exporter instanceof NoOpSpanExporter);
    }

    @Test
    void getTracer_returnsTracer() {
        io.opentelemetry.api.trace.Tracer otelTracer = tracer.getTracer();
        assertNotNull(otelTracer);
    }

    @Test
    void setAttributes_withNull_doesNotThrow() {
        tracer.setAttributes(null);
    }

    private static class TestableBaseTracer extends BaseTracer {
        protected TestableBaseTracer(String projectName, boolean enableEvaluation, JudgmentSyncClient apiClient,
                ISerializer serializer) {
            super(projectName, enableEvaluation, apiClient, serializer);
        }

        @Override
        public void initialize() {
        }

        @Override
        public boolean forceFlush(int timeoutMillis) {
            return false;
        }

        @Override
        public void shutdown(int timeoutMillis) {
        }
    }
}

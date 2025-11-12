package com.judgmentlabs.judgeval.data;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class ExampleTest {

    @Test
    void constructor_generatesExampleId() {
        Example example = new Example();
        assertNotNull(example.getExampleId());
    }

    @Test
    void constructor_setsCreatedAt() {
        Example example = new Example();
        assertNotNull(example.getCreatedAt());
    }

    @Test
    void constructor_setsNameToNull() {
        Example example = new Example();
        assertNull(example.getName());
    }

    @Test
    void builder_createsExample() {
        Example example = Example.builder().build();
        assertNotNull(example);
    }

    @Test
    void builder_withProperty_setsProperty() {
        Example example = Example.builder()
                .property("key", "value")
                .build();

        assertNotNull(example);
        assertEquals("value", example.getAdditionalProperties().get("key"));
    }

    @Test
    void builder_withName_setsName() {
        Example example = Example.builder()
                .name("test-example")
                .build();

        assertEquals("test-example", example.getName());
    }
}

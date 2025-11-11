package com.judgmentlabs.judgeval.v1.data;

import java.time.Instant;
import java.util.UUID;

/**
 * Represents an evaluation example with arbitrary properties.
 */
public class Example extends com.judgmentlabs.judgeval.internal.api.models.Example {

    public Example() {
        super();
        setExampleId(UUID.randomUUID()
                .toString());
        setCreatedAt(Instant.now()
                .toString());
        setName(null);
    }

    /**
     * Creates a new builder for configuring an Example.
     *
     * @return a new builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for configuring and creating Example instances.
     */
    public static final class Builder {
        private final Example example;

        private Builder() {
            this.example = new Example();
        }

        /**
         * Sets a custom property on the example.
         *
         * @param key
         *            the property key
         * @param value
         *            the property value
         * @return this builder
         */
        public Builder property(String key, Object value) {
            example.setAdditionalProperty(key, value);
            return this;
        }

        /**
         * Sets the name of the example.
         *
         * @param name
         *            the example name
         * @return this builder
         */
        public Builder name(String name) {
            example.setName(name);
            return this;
        }

        /**
         * Builds and returns the configured Example.
         *
         * @return the configured Example
         */
        public Example build() {
            return example;
        }
    }
}

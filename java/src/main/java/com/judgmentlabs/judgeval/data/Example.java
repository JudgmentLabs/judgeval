package com.judgmentlabs.judgeval.data;

import java.time.Instant;
import java.util.UUID;

/**
 * @deprecated Use {@link com.judgmentlabs.judgeval.v1.data.Example} instead.
 */
@Deprecated
public class Example extends com.judgmentlabs.judgeval.internal.api.models.Example {

    public Example() {
        super();
        setExampleId(UUID.randomUUID()
                .toString());
        setCreatedAt(Instant.now()
                .toString());
        setName(null);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private final Example example;

        private Builder() {
            this.example = new Example();
        }

        public Builder property(String key, Object value) {
            example.setAdditionalProperty(key, value);
            return this;
        }

        public Builder name(String name) {
            example.setName(name);
            return this;
        }

        public Example build() {
            return example;
        }
    }
}

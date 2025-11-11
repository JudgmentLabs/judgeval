package com.judgmentlabs.judgeval;

public final class Judgeval extends com.judgmentlabs.judgeval.v1.Judgeval {
    private Judgeval(Builder builder) {
        super(builder);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder extends com.judgmentlabs.judgeval.v1.Judgeval.Builder {
        @Override
        public Judgeval build() {
            return new Judgeval(this);
        }
    }
}

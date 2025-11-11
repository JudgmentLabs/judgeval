package com.judgmentlabs.judgeval.v1.scorers.custom_scorer;

/**
 * Factory for creating custom scorer instances.
 */
public final class CustomScorerFactory {
    public CustomScorerFactory() {
    }

    /**
     * Creates a custom scorer with the specified name.
     *
     * @param name
     *            the scorer name
     * @return the configured custom scorer
     */
    public CustomScorer get(String name) {
        return CustomScorer.builder()
                .name(name)
                .className(name)
                .build();
    }

    /**
     * Creates a custom scorer with the specified name and class name.
     *
     * @param name
     *            the scorer name
     * @param className
     *            the class name
     * @return the configured custom scorer
     */
    public CustomScorer get(String name, String className) {
        return CustomScorer.builder()
                .name(name)
                .className(className)
                .build();
    }
}

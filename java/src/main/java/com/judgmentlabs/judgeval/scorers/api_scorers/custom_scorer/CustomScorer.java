package com.judgmentlabs.judgeval.scorers.api_scorers.custom_scorer;

import com.judgmentlabs.judgeval.data.APIScorerType;
import com.judgmentlabs.judgeval.internal.api.models.ScorerConfig;
import com.judgmentlabs.judgeval.scorers.APIScorer;

/**
 * Server-hosted custom scorer representation for enqueue payloads.
 * Instances serialize into ExampleEvaluationRun.custom_scorers with score_type
 * "Custom", server_hosted=true, and optional class_name for server routing.
 *
 * @deprecated Use {@link com.judgmentlabs.judgeval.v1.Judgeval} instead.
 * 
 *             <p>
 *             Migration example:
 * 
 *             <pre>{@code
 * // Old way:
 * CustomScorer scorer = CustomScorer.get("my-scorer");
 * 
 * // New way:
 * Judgeval client = Judgeval.builder().build();
 * CustomScorer scorer = client.scorers().customScorer().get("my-scorer");
 * }</pre>
 */
@Deprecated
public class CustomScorer extends APIScorer {
    public CustomScorer() {
        super(APIScorerType.CUSTOM);
    }

    /**
     * Creates a server-hosted custom scorer with the given name.
     *
     * @param name
     *            scorer identifier and default class_name
     * @return configured CustomScorer
     */
    public static CustomScorer get(String name) {
        CustomScorer s = new CustomScorer();
        s.setName(name);
        s.setClassName(name);
        s.setServerHosted(true);
        return s;
    }

    /**
     * Creates a server-hosted custom scorer with explicit class_name for routing.
     *
     * @param name
     *            scorer identifier
     * @param className
     *            server-side scorer class name
     * @return configured CustomScorer
     */
    public static CustomScorer get(String name, String className) {
        CustomScorer s = get(name);
        s.setClassName(className);
        return s;
    }

    @Override
    /**
     * Not used for server-hosted custom scorers.
     * 
     * @deprecated never returns; always throws
     * @return never returns; always throws
     */
    public ScorerConfig getScorerConfig() {
        throw new UnsupportedOperationException("CustomScorer does not use ScorerConfig");
    }
}

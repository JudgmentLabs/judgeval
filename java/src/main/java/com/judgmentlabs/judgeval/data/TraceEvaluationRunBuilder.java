package com.judgmentlabs.judgeval.data;

import java.time.Instant;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

import com.judgmentlabs.judgeval.Env;
import com.judgmentlabs.judgeval.internal.api.models.ScorerConfig;
import com.judgmentlabs.judgeval.internal.api.models.TraceEvaluationRun;
import com.judgmentlabs.judgeval.scorers.BaseScorer;
import com.judgmentlabs.judgeval.scorers.api_scorers.custom_scorer.CustomScorer;

/**
 * @deprecated Replaced by
 *             com.judgmentlabs.judgeval.v1.data.TraceEvaluationRunBuilder
 */
@Deprecated
public class TraceEvaluationRunBuilder {
    private String                 projectName;
    private String                 evalName;
    private String                 model;
    private String                 traceId;
    private String                 spanId;
    private final List<BaseScorer> scorers = new ArrayList<>();

    public TraceEvaluationRunBuilder projectName(String v) {
        this.projectName = v;
        return this;
    }

    public TraceEvaluationRunBuilder evalName(String v) {
        this.evalName = v;
        return this;
    }

    public TraceEvaluationRunBuilder model(String v) {
        this.model = v;
        return this;
    }

    public TraceEvaluationRunBuilder trace(String traceId, String spanId) {
        this.traceId = traceId;
        this.spanId = spanId;
        return this;
    }

    public TraceEvaluationRunBuilder addScorer(BaseScorer v) {
        this.scorers.add(v);
        return this;
    }

    public TraceEvaluationRun build() {
        TraceEvaluationRun run = new TraceEvaluationRun();
        run.setProjectName(projectName);
        run.setEvalName(evalName);
        run.setModel(model != null ? model : Env.JUDGMENT_DEFAULT_GPT_MODEL);
        List<ScorerConfig> judgment = new ArrayList<>();
        List<com.judgmentlabs.judgeval.internal.api.models.BaseScorer> custom = new ArrayList<>();
        for (BaseScorer s : scorers) {
            if (s instanceof CustomScorer)
                custom.add((com.judgmentlabs.judgeval.internal.api.models.BaseScorer) s);
            else
                judgment.add(s.getScorerConfig());
        }
        run.setJudgmentScorers(judgment);
        run.setCustomScorers(custom);
        run.setTraceAndSpanIds(List.of(List.of(traceId, spanId)));
        run.setIsOffline(false);
        run.setIsBucketRun(false);
        run.setId(UUID.randomUUID().toString());
        run.setCreatedAt(Instant.now().atOffset(ZoneOffset.UTC).format(DateTimeFormatter.ISO_INSTANT));
        return run;
    }
}

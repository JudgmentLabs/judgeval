package com.judgmentlabs.judgeval.exceptions;

public class JudgmentTestError extends RuntimeException {
    public JudgmentTestError(String message) {
        super(message);
    }

    public JudgmentTestError(String message, Throwable cause) {
        super(message, cause);
    }
}

package com.judgmentlabs.judgeval.exceptions;

public class JudgmentRuntimeError extends RuntimeException {
    public JudgmentRuntimeError(String message) {
        super(message);
    }

    public JudgmentRuntimeError(String message, Throwable cause) {
        super(message, cause);
    }
}

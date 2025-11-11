package com.judgmentlabs.judgeval.exceptions;

public class JudgmentAPIError extends RuntimeException {
    private final int    statusCode;
    private final String detail;

    public JudgmentAPIError(int statusCode, String detail) {
        super(String.format("%d: %s", statusCode, detail));
        this.statusCode = statusCode;
        this.detail = detail;
    }

    public int getStatusCode() {
        return statusCode;
    }

    public String getDetail() {
        return detail;
    }
}

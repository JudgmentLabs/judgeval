package com.judgmentlabs.judgeval;

public final class Env {
    private Env() {
    }

    public static final String JUDGMENT_API_KEY           = getEnvVar("JUDGMENT_API_KEY");
    public static final String JUDGMENT_ORG_ID            = getEnvVar("JUDGMENT_ORG_ID");
    public static final String JUDGMENT_API_URL           = getEnvVar("JUDGMENT_API_URL",
            "https://api.judgmentlabs.ai");
    public static final String JUDGMENT_DEFAULT_GPT_MODEL = getEnvVar("JUDGMENT_DEFAULT_GPT_MODEL", "gpt-4.1");
    public static final String JUDGMENT_NO_COLOR          = getEnvVar("JUDGMENT_NO_COLOR");
    public static final String JUDGMENT_LOG_LEVEL         = getEnvVar("JUDGMENT_LOG_LEVEL", "warn");

    private static String getEnvVar(String varName) {
        return getEnvVar(varName, null);
    }

    private static String getEnvVar(String varName, String defaultValue) {
        String value = System.getenv(varName);
        return value != null ? value : defaultValue;
    }
}

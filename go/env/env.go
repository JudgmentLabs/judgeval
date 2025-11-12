package env

import "os"

var (
	JudgmentAPIKey          = getEnvVar("JUDGMENT_API_KEY")
	JudgmentOrgID           = getEnvVar("JUDGMENT_ORG_ID")
	JudgmentAPIURL          = getEnvVar("JUDGMENT_API_URL", "https://api.judgmentlabs.ai")
	JudgmentDefaultGPTModel = getEnvVar("JUDGMENT_DEFAULT_GPT_MODEL", "gpt-4.1")
	JudgmentNoColor         = getEnvVar("JUDGMENT_NO_COLOR")
	JudgmentLogLevel        = getEnvVar("JUDGMENT_LOG_LEVEL", "warn")
)

func getEnvVar(varName string, defaultValue ...string) string {
	value := os.Getenv(varName)
	if value != "" {
		return value
	}
	if len(defaultValue) > 0 {
		return defaultValue[0]
	}
	return ""
}

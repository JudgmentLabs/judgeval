package judgeval

type clientConfig struct {
	apiKey string
	orgID  string
	apiURL string
}

type Option interface {
	apply(*clientConfig)
}

type optionFunc func(*clientConfig)

func (f optionFunc) apply(c *clientConfig) {
	f(c)
}

func WithAPIKey(key string) Option {
	return optionFunc(func(c *clientConfig) {
		c.apiKey = key
	})
}

func WithOrganizationID(id string) Option {
	return optionFunc(func(c *clientConfig) {
		c.orgID = id
	})
}

func WithAPIURL(url string) Option {
	return optionFunc(func(c *clientConfig) {
		c.apiURL = url
	})
}

func Bool(v bool) *bool {
	return &v
}

func String(v string) *string {
	return &v
}

func Float(v float64) *float64 {
	return &v
}

func Int(v int) *int {
	return &v
}

func getBool(ptr *bool, defaultVal bool) bool {
	if ptr == nil {
		return defaultVal
	}
	return *ptr
}

func getString(ptr *string, defaultVal string) string {
	if ptr == nil {
		return defaultVal
	}
	return *ptr
}

func getFloat(ptr *float64, defaultVal float64) float64 {
	if ptr == nil {
		return defaultVal
	}
	return *ptr
}

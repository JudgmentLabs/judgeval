package logger

import (
	"fmt"
	"os"
	"strings"
	"time"
)

const (
	RESET  = "\033[0m"
	RED    = "\033[31m"
	YELLOW = "\033[33m"
	GRAY   = "\033[90m"
)

type Level int

const (
	DEBUG Level = iota
	INFO
	WARNING
	ERROR
	CRITICAL
)

var levelNames = map[Level]string{
	DEBUG:    "DEBUG",
	INFO:     "INFO",
	WARNING:  "WARNING",
	ERROR:    "ERROR",
	CRITICAL: "CRITICAL",
}

var levelColors = map[Level]string{
	DEBUG:    GRAY,
	INFO:     GRAY,
	WARNING:  YELLOW,
	ERROR:    RED,
	CRITICAL: RED,
}

var level = WARNING
var useColor = true

func init() {
	noColor := os.Getenv("JUDGMENT_NO_COLOR")
	useColor = noColor == ""

	if levelStr := os.Getenv("JUDGMENT_LOG_LEVEL"); levelStr != "" {
		SetLevelFromString(levelStr)
	}
}

func SetLevel(l Level) {
	level = l
}

func SetUseColor(enableColor bool) {
	useColor = enableColor
}

func SetLevelFromString(levelStr string) {
	switch strings.ToLower(levelStr) {
	case "debug":
		level = DEBUG
	case "info":
		level = INFO
	case "warning", "warn":
		level = WARNING
	case "error":
		level = ERROR
	case "critical":
		level = CRITICAL
	default:
		level = WARNING
	}
}

func log(l Level, format string, args ...interface{}) {
	if l < level {
		return
	}

	message := fmt.Sprintf(format, args...)
	timestamp := time.Now().Format("2006-01-02 15:04:05")
	formattedMessage := fmt.Sprintf("%s - judgeval - %s - %s", timestamp, levelNames[l], message)

	if useColor {
		formattedMessage = levelColors[l] + formattedMessage + RESET
	}

	fmt.Println(formattedMessage)
}

func Debug(format string, args ...interface{}) {
	log(DEBUG, format, args...)
}

func Info(format string, args ...interface{}) {
	log(INFO, format, args...)
}

func Warning(format string, args ...interface{}) {
	log(WARNING, format, args...)
}

func Error(format string, args ...interface{}) {
	log(ERROR, format, args...)
}

func Critical(format string, args ...interface{}) {
	log(CRITICAL, format, args...)
}

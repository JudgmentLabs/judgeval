package com.judgmentlabs.judgeval.utils;

import java.io.PrintStream;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.concurrent.atomic.AtomicBoolean;

import com.judgmentlabs.judgeval.Env;

public class Logger {
    private static final String            RESET          = "\033[0m";
    private static final String            RED            = "\033[31m";
    private static final String            YELLOW         = "\033[33m";
    private static final String            GRAY           = "\033[90m";

    private static final DateTimeFormatter DATE_FORMATTER = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    public enum Level {
        DEBUG(0, GRAY),
        INFO(1, GRAY),
        WARNING(2, YELLOW),
        ERROR(3, RED),
        CRITICAL(4, RED);

        private final int    value;
        private final String color;

        Level(int value, String color) {
            this.value = value;
            this.color = color;
        }

        public int getValue() {
            return value;
        }

        public String getColor() {
            return color;
        }
    }

    private static final AtomicBoolean initialized  = new AtomicBoolean(false);
    private static Level               currentLevel = Level.WARNING;
    private static boolean             useColor     = true;
    private static PrintStream         output       = System.out;

    private static void initialize() {
        if (initialized.compareAndSet(false, true)) {
            String noColor = Env.JUDGMENT_NO_COLOR;
            useColor = noColor == null && System.console() != null;

            String logLevel = Env.JUDGMENT_LOG_LEVEL;
            if (logLevel != null) {
                loadLogLevel(logLevel.toLowerCase());
            }
        }
    }

    private static void loadLogLevel(String logLevel) {
        switch (logLevel) {
            case "debug":
                currentLevel = Level.DEBUG;
                break;
            case "info":
                currentLevel = Level.INFO;
                break;
            case "warning":
            case "warn":
                currentLevel = Level.WARNING;
                break;
            case "error":
                currentLevel = Level.ERROR;
                break;
            case "critical":
                currentLevel = Level.CRITICAL;
                break;
            default:
                currentLevel = Level.WARNING;
                break;
        }
    }

    public static void setLevel(Level level) {
        currentLevel = level;
    }

    public static void setUseColor(boolean useColor) {
        Logger.useColor = useColor;
    }

    public static void setOutput(PrintStream output) {
        Logger.output = output;
    }

    private static void log(Level level, String message) {
        initialize();

        if (level.getValue() < currentLevel.getValue()) {
            return;
        }

        String timestamp = LocalDateTime.now()
                .format(DATE_FORMATTER);
        String formattedMessage = String.format("%s - judgeval - %s - %s", timestamp, level.name(), message);

        if (useColor) {
            formattedMessage = level.getColor() + formattedMessage + RESET;
        }

        output.println(formattedMessage);
    }

    public static void debug(String message) {
        log(Level.DEBUG, message);
    }

    public static void info(String message) {
        log(Level.INFO, message);
    }

    public static void warning(String message) {
        log(Level.WARNING, message);
    }

    public static void error(String message) {
        log(Level.ERROR, message);
    }

    public static void critical(String message) {
        log(Level.CRITICAL, message);
    }
}

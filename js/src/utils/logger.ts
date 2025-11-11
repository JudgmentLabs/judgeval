import { JUDGMENT_LOG_LEVEL } from "../env";

export class Logger {
  private static readonly RESET = "\x1b[0m";
  private static readonly RED = "\x1b[31m";
  private static readonly YELLOW = "\x1b[33m";
  private static readonly GRAY = "\x1b[90m";

  public static Level = {
    DEBUG: 0,
    INFO: 1,
    WARNING: 2,
    ERROR: 3,
    CRITICAL: 4,
  } as const;

  private static initialized = false;
  private static currentLevel: number = Logger.Level.WARNING;
  private static useColor = true;

  private static initialize(): void {
    if (!Logger.initialized) {
      const noColor = process.env.JUDGMENT_NO_COLOR;
      Logger.useColor = !noColor && process.stdout.isTTY;

      const logLevel = JUDGMENT_LOG_LEVEL.toLowerCase();
      if (logLevel) {
        const levelMap: Record<string, number> = {
          debug: Logger.Level.DEBUG,
          info: Logger.Level.INFO,
          warning: Logger.Level.WARNING,
          warn: Logger.Level.WARNING,
          error: Logger.Level.ERROR,
          critical: Logger.Level.CRITICAL,
        };
        Logger.currentLevel = levelMap[logLevel] ?? Logger.Level.WARNING;
      }

      Logger.initialized = true;
    }
  }

  public static setLevel(level: number): void {
    Logger.currentLevel = level;
  }

  public static setUseColor(useColor: boolean): void {
    Logger.useColor = useColor;
  }

  private static log(level: number, message: string): void {
    Logger.initialize();

    if (level < Logger.currentLevel) {
      return;
    }

    const timestamp = new Date()
      .toISOString()
      .replace("T", " ")
      .substring(0, 19);
    const levelName =
      Object.keys(Logger.Level).find(
        (key) => Logger.Level[key as keyof typeof Logger.Level] === level,
      ) ?? "UNKNOWN";
    let formattedMessage = `${timestamp} - judgeval - ${levelName} - ${message}`;

    if (Logger.useColor) {
      const color =
        level === Logger.Level.DEBUG || level === Logger.Level.INFO
          ? Logger.GRAY
          : level === Logger.Level.WARNING
            ? Logger.YELLOW
            : Logger.RED;
      formattedMessage = `${color}${formattedMessage}${Logger.RESET}`;
    }

    const output =
      level >= Logger.Level.ERROR ? process.stderr : process.stdout;
    output.write(formattedMessage + "\n");
  }

  public static debug(message: string): void {
    Logger.log(Logger.Level.DEBUG, message);
  }

  public static info(message: string): void {
    Logger.log(Logger.Level.INFO, message);
  }

  public static warning(message: string): void {
    Logger.log(Logger.Level.WARNING, message);
  }

  public static warn(message: string): void {
    Logger.log(Logger.Level.WARNING, message);
  }

  public static error(message: string): void {
    Logger.log(Logger.Level.ERROR, message);
  }

  public static critical(message: string): void {
    Logger.log(Logger.Level.CRITICAL, message);
  }
}

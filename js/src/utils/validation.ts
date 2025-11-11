/**
 * Validates that a value is not null or undefined and returns it with proper typing.
 * Throws an error with a custom message if the value is null or undefined.
 *
 * @param value - The value to check
 * @param message - The error message to throw if value is null/undefined
 * @returns The value with non-null typing
 * @throws Error if value is null or undefined
 */
export function requireNonNull<T>(
  value: T | null | undefined,
  message: string,
): T {
  if (value === null || value === undefined) {
    throw new Error(message);
  }
  return value;
}

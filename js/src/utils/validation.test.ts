import { describe, expect, test } from "bun:test";
import { requireNonNull } from "./validation";

describe("requireNonNull", () => {
  test("returns valid string value", () => {
    expect(requireNonNull("test", "error")).toBe("test");
  });

  test("returns valid number value", () => {
    expect(requireNonNull(42, "error")).toBe(42);
    expect(requireNonNull(0, "error")).toBe(0);
    expect(requireNonNull(-1, "error")).toBe(-1);
  });

  test("returns valid object value", () => {
    const obj = { key: "value" };
    expect(requireNonNull(obj, "error")).toBe(obj);
  });

  test("returns valid array value", () => {
    const arr = [1, 2, 3];
    expect(requireNonNull(arr, "error")).toBe(arr);
  });

  test("returns falsy values that are not null or undefined", () => {
    expect(requireNonNull(0, "error")).toBe(0);
    expect(requireNonNull("", "error")).toBe("");
    expect(requireNonNull(false, "error")).toBe(false);
  });

  test("throws error with custom message when value is null", () => {
    expect(() => requireNonNull(null, "Custom null error")).toThrow(
      "Custom null error",
    );
  });

  test("throws error with custom message when value is undefined", () => {
    expect(() => {
      requireNonNull(undefined, "Custom undefined error");
    }).toThrow("Custom undefined error");
  });

  test("preserves type information", () => {
    const value: string | null = "test";
    const result = requireNonNull(value, "error");
    expect(typeof result).toBe("string");
  });
});

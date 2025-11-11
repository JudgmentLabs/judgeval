import { describe, expect, test } from "bun:test";
import {
  expectApiKey,
  expect as expectGuard,
  expectOrganizationId,
} from "./guards";

describe("expect", () => {
  test("returns value when valid", () => {
    expect(expectGuard("test", "error")).toBe("test");
    expect(expectGuard(42, "error")).toBe(42);
    expect(expectGuard({ key: "value" }, "error")).toEqual({ key: "value" });
    expect(expectGuard([1, 2, 3], "error")).toEqual([1, 2, 3]);
  });

  test("throws error when value is null", () => {
    expect(() => expectGuard(null, "Value is null")).toThrow("Value is null");
  });

  test("throws error when value is undefined", () => {
    expect(() => {
      expectGuard(undefined, "Value is undefined");
    }).toThrow("Value is undefined");
  });

  test("returns falsy values that are not null or undefined", () => {
    expect(expectGuard(0, "error")).toBe(0);
    expect(expectGuard("", "error")).toBe("");
    expect(expectGuard(false, "error")).toBe(false);
  });
});

describe("expectApiKey", () => {
  test("returns valid API key", () => {
    expect(expectApiKey("sk-1234567890")).toBe("sk-1234567890");
    expect(expectApiKey("valid_key")).toBe("valid_key");
  });

  test("throws error when API key is null", () => {
    expect(() => expectApiKey(null)).toThrow("API key is required");
  });

  test("throws error when API key is undefined", () => {
    expect(() => expectApiKey(undefined)).toThrow("API key is required");
  });

  test("throws error when API key is empty string", () => {
    expect(() => expectApiKey("")).toThrow("API key is required");
  });

  test("throws error when API key is whitespace only", () => {
    expect(() => expectApiKey("   ")).toThrow("API key is required");
    expect(() => expectApiKey("\t\n")).toThrow("API key is required");
  });
});

describe("expectOrganizationId", () => {
  test("returns valid organization ID", () => {
    expect(expectOrganizationId("org-123")).toBe("org-123");
    expect(expectOrganizationId("valid_org_id")).toBe("valid_org_id");
  });

  test("throws error when organization ID is null", () => {
    expect(() => expectOrganizationId(null)).toThrow(
      "Organization ID is required",
    );
  });

  test("throws error when organization ID is undefined", () => {
    expect(() => expectOrganizationId(undefined)).toThrow(
      "Organization ID is required",
    );
  });

  test("throws error when organization ID is empty string", () => {
    expect(() => expectOrganizationId("")).toThrow(
      "Organization ID is required",
    );
  });

  test("throws error when organization ID is whitespace only", () => {
    expect(() => expectOrganizationId("   ")).toThrow(
      "Organization ID is required",
    );
    expect(() => expectOrganizationId("\t\n")).toThrow(
      "Organization ID is required",
    );
  });
});

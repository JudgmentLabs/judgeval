import { describe, expect, test } from "bun:test";
import { parseFunctionArgs } from "./annotate";

describe("parseFunctionArgs", () => {
  test("parses regular function with named parameters", () => {
    function testFunc(a: string, _b: number, _c: boolean) {
      return a;
    }
    expect(parseFunctionArgs(testFunc)).toEqual(["a", "_b", "_c"]);
  });

  test("parses arrow function with single parameter", () => {
    const testFunc = (x: number) => x * 2;
    expect(parseFunctionArgs(testFunc)).toEqual(["x"]);
  });

  test("parses arrow function with multiple parameters", () => {
    const testFunc = (a: string, _b: number, _c: boolean) => a;
    expect(parseFunctionArgs(testFunc)).toEqual(["a", "_b", "_c"]);
  });

  test("parses function with no parameters", () => {
    function testFunc() {
      return 42;
    }
    expect(parseFunctionArgs(testFunc)).toEqual([]);
  });

  test("parses arrow function with no parameters", () => {
    const testFunc = () => 42;
    expect(parseFunctionArgs(testFunc)).toEqual([]);
  });

  test("parses function with comments in signature", () => {
    function testFunc(
      a: string /* comment */,
      _b: number, // another comment
      _c: boolean,
    ) {
      return a;
    }
    expect(parseFunctionArgs(testFunc)).toEqual(["a", "_b", "_c"]);
  });

  test("parses function with default parameters", () => {
    function testFunc(a = "default", _b = 42) {
      return a;
    }
    expect(parseFunctionArgs(testFunc)).toEqual(['a = "default"', "_b = 42"]);
  });

  test("parses function with destructured parameters", () => {
    function testFunc({ x, y }: { x: number; y: number }) {
      return x + y;
    }
    const result = parseFunctionArgs(testFunc);
    expect(result).toEqual(["{ x", "y }"]);
  });

  test("parses function with rest parameters", () => {
    function testFunc(a: string, ..._rest: number[]) {
      return a;
    }
    expect(parseFunctionArgs(testFunc)).toEqual(["a", "..._rest"]);
  });

  test("handles function with whitespace in parameters", () => {
    function testFunc(a: string, _b: number) {
      return a;
    }
    expect(parseFunctionArgs(testFunc)).toEqual(["a", "_b"]);
  });

  test("parses arrow function without parentheses", () => {
    const testFunc = (x: number) => x * 2;
    expect(parseFunctionArgs(testFunc)).toEqual(["x"]);
  });
});

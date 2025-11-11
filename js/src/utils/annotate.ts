/* eslint-disable */
/// Adopted from https://github.com/angular/angular.js/blob/master/src/auto/injector.js

const ARROW_ARG = /^([^(]+?)=>/;
const FN_ARGS = /^[^(]*\(\s*([^)]*)\)/m;
const FN_ARG_SPLIT = /,/;
const FN_ARG = /^\s*(_?)(\S+?)\1\s*$/;
const STRIP_COMMENTS = /((\/\/.*$)|(\/\*[\s\S]*?\*\/))/gm;

function stringifyFn(fn: Function): string {
  return Function.prototype.toString.call(fn);
}

function extractArgs(fn: Function): RegExpMatchArray | null {
  const fnText = stringifyFn(fn).replace(STRIP_COMMENTS, "");
  return fnText.match(ARROW_ARG) || fnText.match(FN_ARGS);
}

export function parseFunctionArgs(fn: Function): string[] {
  const args = extractArgs(fn);
  if (!args || !args[1]) {
    return [];
  }

  return args[1]
    .split(FN_ARG_SPLIT)
    .map((arg) => {
      const match = arg.replace(FN_ARG, (all, underscore, name) => name);
      return match.trim();
    })
    .filter((name) => name.length > 0);
}

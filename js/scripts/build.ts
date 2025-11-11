#!/usr/bin/env bun

import type { BuildConfig, BuildOutput } from "bun";
import { build } from "bun";

const isDev = process.argv.includes("--dev");
const isProduction = !isDev;

function handleBuildResult(result: BuildOutput, target: string) {
  if (!result.success) {
    console.error(`Build failed for target '${target}':`);
    result.logs.forEach((log) => {
      const level = log.level === "error" ? "ERROR" : log.level.toUpperCase();
      console.error(`  [${level}] ${log.message}`);
      if (log.position) {
        console.error(
          `    at ${log.position.file}:${log.position.line}:${log.position.column}`
        );
      }
    });
    process.exit(1);
  }

  console.log(
    `✓ Built ${target} bundle: ${result.outputs.map((o) => o.path).join(", ")}`
  );
}

async function buildLib() {
  const config: BuildConfig = {
    entrypoints: ["./src/index.ts", "./src/v1/index.ts"],
    outdir: "./dist",
    target: "node",
    format: "esm",
    external: ["@opentelemetry/*"],
    minify: isProduction,
    sourcemap: isProduction ? "linked" : "inline",
    naming: { entry: "[dir]/[name].mjs" },
  };

  const result = await build(config);
  handleBuildResult(result, "lib");
}

async function buildCjs() {
  const config: BuildConfig = {
    entrypoints: ["./src/index.ts", "./src/v1/index.ts"],
    outdir: "./dist",
    target: "node",
    format: "cjs",
    external: ["@opentelemetry/*"],
    minify: isProduction,
    sourcemap: isProduction ? "linked" : "inline",
    naming: { entry: "[dir]/[name].cjs" },
  };

  const result = await build(config);
  handleBuildResult(result, "cjs");
}

async function buildUmd() {
  const config: BuildConfig = {
    entrypoints: ["./src/index.ts"],
    outdir: "./dist",
    target: "browser",
    format: "iife",
    external: ["@opentelemetry/*"],
    minify: isProduction,
    sourcemap: isProduction ? "linked" : "inline",
    naming: { entry: "index.umd.js" },
  };

  const result = await build(config);
  handleBuildResult(result, "umd");
}

async function main() {
  const args = process.argv.slice(2).filter((arg) => !arg.startsWith("--"));
  const targetsToBuild = args.length > 0 ? args : ["lib", "cjs", "umd"];

  const buildPromises = targetsToBuild.map((target) => {
    switch (target) {
      case "lib":
        return buildLib();
      case "cjs":
        return buildCjs();
      case "umd":
        return buildUmd();
      default:
        console.error(`Unknown target: ${target}`);
        console.error("Available targets: lib, cjs, umd");
        process.exit(1);
    }
  });

  await Promise.all(buildPromises);
}

main().catch(console.error);

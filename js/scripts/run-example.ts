const exampleName = process.argv[2];

if (!exampleName) {
  console.error("Usage: bun run example <example-name>");
  process.exit(1);
}

const examplePath = `examples/${exampleName}`;

process.chdir(examplePath);

const { spawnSync } = require("child_process");
const result = spawnSync("npm", ["run", "start"], {
  stdio: "inherit",
  cwd: process.cwd(),
});

process.exit(result.status ?? 0);

#!/usr/bin/env node

import * as fs from "fs";
import * as http from "http";
import * as https from "https";
import * as path from "path";

const JUDGEVAL_PATHS = [
  "/log_eval_results/",
  "/fetch_experiment_run/",
  "/add_to_run_eval_queue/examples",
  "/add_to_run_eval_queue/traces",
  "/get_evaluation_status/",
  "/save_scorer/",
  "/fetch_scorers/",
  "/scorer_exists/",
  "/projects/resolve/",
];

const HTTP_METHODS = new Set(["GET", "POST", "PUT", "PATCH", "DELETE"]);
const SUCCESS_STATUS_CODES = new Set(["200", "201"]);
const SCHEMA_REF_PREFIX = "#/components/schemas/";

interface QueryParam {
  name: string;
  required: boolean;
  type: string;
}

interface MethodInfo {
  name: string;
  path: string;
  method: string;
  request_type?: string;
  query_params: QueryParam[];
  response_type: string;
}

interface OpenAPISpec {
  paths: Record<string, Record<string, any>>;
  components: {
    schemas: Record<string, any>;
  };
}

function resolveRef(ref: string): string {
  if (!ref.startsWith(SCHEMA_REF_PREFIX)) {
    throw new Error(`Reference must start with ${SCHEMA_REF_PREFIX}`);
  }
  return ref.replace(SCHEMA_REF_PREFIX, "");
}

function toCamelCase(name: string): string {
  const parts = name.replace(/-/g, "_").split("_");
  return (
    parts[0] +
    parts
      .slice(1)
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join("")
  );
}

function toClassName(name: string): string {
  const camelCase = toCamelCase(name);
  return camelCase.charAt(0).toUpperCase() + camelCase.slice(1);
}

function getMethodNameFromPath(path: string, method: string): string {
  const cleanPath = path
    .replace(/^\//, "")
    .replace(/\//g, "_")
    .replace(/-/g, "_");
  return toCamelCase(cleanPath);
}

function getQueryParameters(operation: any): QueryParam[] {
  return (operation.parameters || [])
    .filter((param: any) => param.in === "query")
    .map((param: any) => ({
      name: param.name,
      required: param.required || false,
      type: param.schema?.type || "string",
    }));
}

function getSchemaFromContent(content: any): string | null {
  if (!content || typeof content !== "object") {
    return null;
  }

  if (content["application/json"]) {
    const schema = content["application/json"].schema || {};
    if (schema.$ref) {
      return resolveRef(schema.$ref);
    }

    if (Object.keys(schema).length === 0) {
      return "EMPTY_SCHEMA";
    }

    if (schema.type) {
      return null; // For now, we only handle $ref schemas
    }
  }

  for (const contentType of Object.keys(content)) {
    if (content[contentType] && content[contentType].schema) {
      const schema = content[contentType].schema;
      if (schema.$ref) {
        return resolveRef(schema.$ref);
      }

      if (Object.keys(schema).length === 0) {
        return "EMPTY_SCHEMA";
      }
    }
  }

  return null;
}

function getRequestSchema(operation: any): string | null {
  const requestBody = operation.requestBody;
  return requestBody ? getSchemaFromContent(requestBody.content || {}) : null;
}

function getResponseSchema(operation: any): string | null {
  const responses = operation.responses || {};
  const SUCCESS_STATUS_CODES = new Set(["200", "201"]);

  for (const statusCode of SUCCESS_STATUS_CODES) {
    if (statusCode in responses) {
      const response = responses[statusCode];
      const result = getSchemaFromContent(response.content || {});
      if (result) return result;
    }
  }

  for (const [statusCode, response] of Object.entries(responses)) {
    if (
      statusCode.startsWith("2") &&
      response &&
      typeof response === "object" &&
      "content" in response
    ) {
      const result = getSchemaFromContent(response.content || {});
      if (result) return result;
    }
  }

  for (const [statusCode, response] of Object.entries(responses)) {
    if (
      !statusCode.startsWith("4") &&
      !statusCode.startsWith("5") &&
      response &&
      typeof response === "object" &&
      "content" in response
    ) {
      const result = getSchemaFromContent(response.content || {});
      if (result) return result;
    }
  }

  return null;
}

function extractDependencies(
  schema: any,
  visited: Set<string> = new Set()
): Set<string> {
  const schemaKey = JSON.stringify(schema, Object.keys(schema).sort());
  if (visited.has(schemaKey)) {
    return new Set();
  }

  visited.add(schemaKey);
  const dependencies = new Set<string>();

  if (schema.$ref) {
    return new Set([resolveRef(schema.$ref)]);
  }

  for (const key of ["anyOf", "oneOf", "allOf"]) {
    if (schema[key]) {
      for (const s of schema[key]) {
        for (const dep of extractDependencies(s, visited)) {
          dependencies.add(dep);
        }
      }
    }
  }

  if (schema.properties) {
    for (const propSchema of Object.values(schema.properties)) {
      for (const dep of extractDependencies(propSchema as any, visited)) {
        dependencies.add(dep);
      }
    }
  }

  if (schema.items) {
    for (const dep of extractDependencies(schema.items, visited)) {
      dependencies.add(dep);
    }
  }

  if (
    schema.additionalProperties &&
    typeof schema.additionalProperties === "object"
  ) {
    for (const dep of extractDependencies(
      schema.additionalProperties,
      visited
    )) {
      dependencies.add(dep);
    }
  }

  return dependencies;
}

function findUsedSchemas(spec: OpenAPISpec): Set<string> {
  const usedSchemas = new Set<string>();
  const schemas = spec.components?.schemas || {};
  const HTTP_METHODS = new Set(["GET", "POST", "PUT", "PATCH", "DELETE"]);

  for (const path of JUDGEVAL_PATHS) {
    if (path in spec.paths) {
      for (const [method, operation] of Object.entries(spec.paths[path])) {
        if (HTTP_METHODS.has(method.toUpperCase())) {
          const requestSchema = getRequestSchema(operation);
          const responseSchema = getResponseSchema(operation);

          if (requestSchema) usedSchemas.add(requestSchema);
          if (responseSchema) usedSchemas.add(responseSchema);
        }
      }
    }
  }

  let changed = true;
  while (changed) {
    changed = false;
    const newSchemas = new Set<string>();

    for (const schemaName of usedSchemas) {
      if (schemaName in schemas) {
        const deps = extractDependencies(schemas[schemaName]);
        for (const dep of deps) {
          if (dep in schemas && !usedSchemas.has(dep)) {
            newSchemas.add(dep);
            changed = true;
          }
        }
      }
    }

    for (const schema of newSchemas) {
      usedSchemas.add(schema);
    }
  }

  return usedSchemas;
}

function getTypeScriptType(
  schema: any,
  referencedTypes: Set<string> = new Set(),
  visited: Set<string> = new Set()
): string {
  if (!schema) {
    return "any";
  }

  if (schema.$ref) {
    const typeName = toClassName(resolveRef(schema.$ref));
    if (visited.has(typeName)) {
      return typeName; // Avoid circular references
    }
    referencedTypes.add(typeName);
    return typeName;
  }

  if (schema.anyOf) {
    const unions = schema.anyOf.map((s: any) =>
      getTypeScriptType(s, referencedTypes, visited)
    );
    return `(${unions.join(" | ")})`;
  }

  if (schema.oneOf) {
    const unions = schema.oneOf.map((s: any) =>
      getTypeScriptType(s, referencedTypes, visited)
    );
    return `(${unions.join(" | ")})`;
  }

  if (schema.allOf) {
    const intersections = schema.allOf.map((s: any) =>
      getTypeScriptType(s, referencedTypes, visited)
    );
    return `(${intersections.join(" & ")})`;
  }

  if (schema.enum) {
    if (schema.enum.length === 0) {
      return "never";
    }
    const enumValues = schema.enum
      .map((v: any) => (typeof v === "string" ? `"${v}"` : String(v)))
      .join(" | ");
    return enumValues;
  }

  if (!schema.type) {
    if (schema.properties) {
      return generateObjectType(schema, referencedTypes, visited);
    }

    if (schema.items) {
      return `Array<${getTypeScriptType(
        schema.items,
        referencedTypes,
        visited
      )}>`;
    }

    if (schema.additionalProperties === true) {
      return "Record<string, any>";
    } else if (
      schema.additionalProperties &&
      typeof schema.additionalProperties === "object"
    ) {
      return `Record<string, ${getTypeScriptType(
        schema.additionalProperties,
        referencedTypes,
        visited
      )}>`;
    }

    return "any";
  }

  switch (schema.type) {
    case "string":
      if (schema.const) {
        return `"${schema.const}"`;
      }
      return "string";
    case "number":
    case "integer":
      return "number";
    case "boolean":
      return "boolean";
    case "null":
      return "null";
    case "array":
      if (schema.prefixItems) {
        return `[${schema.prefixItems
          .map((item: any) => getTypeScriptType(item, referencedTypes, visited))
          .join(", ")}]`;
      } else if (!schema.items) {
        return "Array<any>";
      }
      return `Array<${getTypeScriptType(
        schema.items,
        referencedTypes,
        visited
      )}>`;
    case "object":
      return generateObjectType(schema, referencedTypes, visited);
    default:
      return "any";
  }
}

function generateObjectType(
  schema: any,
  referencedTypes: Set<string>,
  visited: Set<string>
): string {
  if (!schema.properties) {
    if (schema.additionalProperties === true) {
      return "Record<string, any>";
    } else if (
      schema.additionalProperties &&
      typeof schema.additionalProperties === "object"
    ) {
      return `Record<string, ${getTypeScriptType(
        schema.additionalProperties,
        referencedTypes,
        visited
      )}>`;
    }
    return "{}";
  }

  const required = schema.required || [];
  const propStrings = Object.entries(schema.properties).map(
    ([propName, propSchema]) => {
      const isRequired = required.includes(propName);
      const propType = getTypeScriptType(
        propSchema as any,
        referencedTypes,
        visited
      );
      const safePropName = /^[a-zA-Z_$][a-zA-Z0-9_$]*$/.test(propName)
        ? propName
        : `"${propName}"`;
      return `  ${safePropName}${isRequired ? "" : "?"}: ${propType};`;
    }
  );

  return `{\n${propStrings.join("\n")}\n}`;
}

function generateModelClass(className: string, schema: any): string {
  const referencedTypes = new Set<string>();
  const visited = new Set<string>();
  const lines: string[] = [];

  lines.push("/**");
  lines.push(" * Auto-generated by scripts/generate-client.ts");
  lines.push(" * DO NOT EDIT MANUALLY - This file is generated automatically");
  lines.push(" */");
  lines.push("");

  const tsType = getTypeScriptType(schema, referencedTypes, visited);

  const imports = Array.from(referencedTypes)
    .filter((type) => type !== className)
    .sort();

  if (imports.length > 0) {
    for (const importType of imports) {
      lines.push(`import { ${importType} } from './${importType}';`);
    }
    lines.push("");
  }

  lines.push(`export interface ${className} ${tsType}`);

  return lines.join("\n");
}

async function fetchSpec(specUrl: string): Promise<OpenAPISpec> {
  return new Promise((resolve, reject) => {
    const client = specUrl.startsWith("https") ? https : http;

    client
      .get(specUrl, (res) => {
        let data = "";
        res.on("data", (chunk) => {
          data += chunk;
        });
        res.on("end", () => {
          try {
            resolve(JSON.parse(data));
          } catch (error) {
            reject(new Error(`Failed to parse JSON: ${error}`));
          }
        });
      })
      .on("error", (error) => {
        reject(error);
      });
  });
}

function generateMethodSignature(
  methodName: string,
  requestType: string | null,
  queryParams: QueryParam[],
  responseType: string
): string {
  const params: string[] = [];

  for (const param of queryParams) {
    if (param.required) {
      params.push(`${param.name}: string`);
    }
  }

  if (requestType) {
    params.push(`payload: Models.${requestType}`);
  }

  for (const param of queryParams) {
    if (!param.required) {
      params.push(`${param.name}?: string`);
    }
  }

  const returnType =
    responseType === "void" ? "void" : `Models.${responseType}`;
  return `  async ${methodName}(${params.join(
    ", "
  )}): Promise<${returnType}> {`;
}

function generateMethodBody(
  methodName: string,
  path: string,
  method: string,
  requestType: string | null,
  queryParams: QueryParam[],
  responseType: string
): string {
  const lines: string[] = [];

  if (queryParams.length > 0) {
    lines.push("    const queryParams = new URLSearchParams();");
    for (const param of queryParams) {
      if (param.required) {
        lines.push(`    queryParams.set("${param.name}", ${param.name});`);
      } else {
        lines.push(`    if (${param.name}) {`);
        lines.push(`      queryParams.set("${param.name}", ${param.name});`);
        lines.push(`    }`);
      }
    }
  }

  const queryString =
    queryParams.length > 0
      ? " + (queryParams.toString() ? '?' + queryParams.toString() : '')"
      : "";
  lines.push(`    const url = this.buildUrl("${path}"${queryString});`);

  if (method === "GET" || method === "DELETE") {
    lines.push("    const response = await fetch(url, {");
    lines.push(`      method: "${method}",`);
    lines.push("      headers: this.buildHeaders(),");
    lines.push("    });");
  } else {
    const payloadExpr = requestType ? "payload" : "{}";
    lines.push(`    const response = await fetch(url, {`);
    lines.push(`      method: "${method}",`);
    lines.push("      headers: this.buildHeaders(),");
    lines.push(`      body: JSON.stringify(${payloadExpr}),`);
    lines.push("    });");
  }

  lines.push("");
  lines.push("    if (!response.ok) {");
  lines.push(
    "      throw new Error(`HTTP Error: ${response.status} - ${await response.text()}`);"
  );
  lines.push("    }");
  lines.push("");

  if (responseType === "void") {
    lines.push("    return;");
  } else if (responseType === "any") {
    lines.push("    return await response.json();");
  } else {
    lines.push(`    return await response.json() as Models.${responseType};`);
  }

  return lines.join("\n");
}

function generateClientClass(className: string, methods: MethodInfo[]): string {
  const lines = [
    "/**",
    " * Auto-generated by scripts/generate-client.ts",
    " * DO NOT EDIT MANUALLY - This file is generated automatically",
    " */",
    "import * as Models from './models';",
    "",
    "export class " + className + " {",
    "  private baseUrl: string;",
    "  private apiKey: string;",
    "  private organizationId: string;",
    "",
    "  constructor(baseUrl: string, apiKey: string, organizationId: string) {",
    "    this.baseUrl = baseUrl;",
    "    this.apiKey = apiKey;",
    "    this.organizationId = organizationId;",
    "  }",
    "",
    "  getBaseUrl(): string {",
    "    return this.baseUrl;",
    "  }",
    "",
    "  getApiKey(): string {",
    "    return this.apiKey;",
    "  }",
    "",
    "  getOrganizationId(): string {",
    "    return this.organizationId;",
    "  }",
    "",
    "  private buildUrl(path: string): string {",
    "    return this.baseUrl + path;",
    "  }",
    "",
    "  private buildHeaders(): Record<string, string> {",
    "    if (!this.apiKey || !this.organizationId) {",
    "      throw new Error('API key and organization ID cannot be null');",
    "    }",
    "    return {",
    "      'Content-Type': 'application/json',",
    "      'Authorization': `Bearer ${this.apiKey}`,",
    "      'X-Organization-Id': this.organizationId,",
    "    };",
    "  }",
    "",
  ];

  for (const methodInfo of methods) {
    const signature = generateMethodSignature(
      methodInfo.name,
      methodInfo.request_type || null,
      methodInfo.query_params,
      methodInfo.response_type
    );
    lines.push(signature);

    const body = generateMethodBody(
      methodInfo.name,
      methodInfo.path,
      methodInfo.method,
      methodInfo.request_type || null,
      methodInfo.query_params,
      methodInfo.response_type
    );
    lines.push(body);
    lines.push("  }");
    lines.push("");
  }

  lines.push("}");
  return lines.join("\n");
}

async function generateApiFiles(spec: OpenAPISpec): Promise<void> {
  const usedSchemas = findUsedSchemas(spec);
  const schemas = spec.components?.schemas || {};

  const modelsDir = "src/internal/api/models";
  if (fs.existsSync(modelsDir)) {
    console.error(`Clearing existing models directory: ${modelsDir}`);
    fs.rmSync(modelsDir, { recursive: true, force: true });
  }

  fs.mkdirSync(modelsDir, { recursive: true });

  console.error("Generating schema types...");
  for (const schemaName of usedSchemas) {
    if (schemaName in schemas) {
      const className = toClassName(schemaName);
      const modelClass = generateModelClass(className, schemas[schemaName]);

      fs.writeFileSync(path.join(modelsDir, `${className}.ts`), modelClass);
      console.error(`Generated schema type: ${className}`);
    }
  }

  const modelFiles = fs
    .readdirSync(modelsDir)
    .filter((file) => file.endsWith(".ts"))
    .map((file) => file.replace(".ts", ""));

  const modelsIndex = [
    "/**",
    " * Auto-generated by scripts/generate-client.ts",
    " * DO NOT EDIT MANUALLY - This file is generated automatically",
    " */",
    "",
    ...modelFiles.map((file) => `export * from './models/${file}';`),
    "",
  ].join("\n");

  const apiDir = "src/internal/api";
  fs.mkdirSync(apiDir, { recursive: true });
  fs.writeFileSync(path.join(apiDir, "models.ts"), modelsIndex);
  console.error(`Generated: ${apiDir}/models.ts`);

  const filteredPaths = Object.fromEntries(
    Object.entries(spec.paths).filter(([path]) => JUDGEVAL_PATHS.includes(path))
  );

  for (const path of JUDGEVAL_PATHS) {
    if (!(path in spec.paths)) {
      console.error(`Path ${path} not found in OpenAPI spec`);
    }
  }

  const methods: MethodInfo[] = [];
  for (const [path, pathData] of Object.entries(filteredPaths)) {
    for (const [method, operation] of Object.entries(pathData)) {
      if (HTTP_METHODS.has(method.toUpperCase())) {
        const methodName = getMethodNameFromPath(path, method.toUpperCase());
        const requestSchema = getRequestSchema(operation);
        const responseSchema = getResponseSchema(operation);
        const queryParams = getQueryParameters(operation);

        console.error(
          `${methodName} ${requestSchema} ${responseSchema} ${JSON.stringify(
            queryParams
          )}`
        );

        const methodInfo: MethodInfo = {
          name: methodName,
          path,
          method: method.toUpperCase(),
          request_type: requestSchema ? toClassName(requestSchema) : undefined,
          query_params: queryParams,
          response_type:
            responseSchema === "EMPTY_SCHEMA"
              ? "void"
              : responseSchema
                ? toClassName(responseSchema)
                : "any",
        };
        methods.push(methodInfo);
      }
    }
  }

  const clientClass = generateClientClass("JudgmentApiClient", methods);
  fs.writeFileSync(path.join(apiDir, "index.ts"), clientClass);
  console.error(`Generated: ${apiDir}/index.ts`);
}

async function main(): Promise<void> {
  const specFile = process.argv[2] || "http://localhost:8000/openapi.json";

  try {
    let spec: OpenAPISpec;

    if (specFile.startsWith("http")) {
      spec = await fetchSpec(specFile);
    } else {
      const specData = fs.readFileSync(specFile, "utf8");
      spec = JSON.parse(specData);
    }

    await generateApiFiles(spec);
  } catch (error) {
    console.error(`Error generating API client: ${error}`);
    process.exit(1);
  }
}

if (require.main === module) {
  main().catch(console.error);
}

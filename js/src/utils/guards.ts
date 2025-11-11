export function expect<T>(value: T | null | undefined, message: string): T {
  if (value === null || value === undefined) {
    throw new Error(message);
  }
  return value;
}

export function expectApiKey(apiKey: string | null | undefined): string {
  if (!apiKey || apiKey.trim() === "") {
    throw new Error("API key is required");
  }
  return apiKey;
}

export function expectOrganizationId(
  organizationId: string | null | undefined,
): string {
  if (!organizationId || organizationId.trim() === "") {
    throw new Error("Organization ID is required");
  }
  return organizationId;
}

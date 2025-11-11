import { Example as ExampleModel } from "../internal/api/models/Example";

/**
 * @deprecated Use v1 Example class instead. This type will be removed in a future version.
 * Import from 'judgeval/v1' instead.
 */
export type Example<
  T extends Record<string, unknown> = Record<string, unknown>,
> = ExampleModel & T & Record<string, unknown>;

/**
 * @deprecated Use v1 Example.builder() instead. This function will be removed in a future version.
 * Import from 'judgeval/v1' instead.
 */
export function Example<T extends Record<string, unknown>>(
  properties: T,
): Example<T> {
  const example: Example<T> = {
    example_id: undefined,
    created_at: new Date().toISOString(),
    name: null,
    ...properties,
  } as Example<T>;

  return example;
}

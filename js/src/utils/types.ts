export type HasRequiredKeys<
  T,
  K extends readonly string[],
> = K extends readonly (infer U)[]
  ? U extends string
    ? T extends Record<U, unknown>
      ? true
      : false
    : false
  : false;

export type KeysOf<T extends readonly string[]> = T[number];

export type RecordWithKeys<K extends readonly string[]> = Record<
  KeysOf<K>,
  unknown
>;

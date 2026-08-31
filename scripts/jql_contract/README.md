# Public JQL code-generation inputs

These files are public-safe snapshots of the canonical JQL contracts in
`JudgmentLabs/judgment-mono`:

- `jql-ir.openapi.json` contains only the public JQL IR schema-reference closure.
- `public-openapi.json` is judgeval-server's public JQL transport contract.

Python package builds regenerate the checked-in contract, query-root, and
transport modules under `src/judgeval/jql` from these snapshots. Pull-request
and release CI separately regenerate from `judgment-mono/main` and fail if the
Python SDK output differs.

After an intentional upstream contract change, refresh the snapshots with:

```sh
python scripts/generate_jql.py --sync \
  ../judgment-mono/services/data-access-service/openapi.json \
  ../judgment-mono/services/judgeval-server/openapi.public-jql.json
```

from __future__ import annotations

from judgeval.v1 import Judgeval

client = Judgeval()

print("=== Testing Prompt Versioning ===\n")

print("1. Creating initial prompt...")
prompt_v1 = client.prompts.create(
    project_name="test-project",
    name="greeting",
    prompt="Hello {{name}}, welcome to {{place}}!",
    tags=["v1.0"],
)
print(f"   Created: commit_id={prompt_v1.commit_id}, tags={prompt_v1.tags}")
print(f"   Compile test: {prompt_v1.compile(name='Alice', place='Wonderland')}\n")

print("2. Creating second version with different content...")
prompt_v2 = client.prompts.create(
    project_name="test-project",
    name="greeting",
    prompt="Hi {{name}}! How are you today? You're in {{place}}.",
    tags=["v2.0", "latest"],
)
print(f"   Created: commit_id={prompt_v2.commit_id}, tags={prompt_v2.tags}")
print(f"   Parent: {prompt_v2.parent_commit_id}")
print(f"   Compile test: {prompt_v2.compile(name='Bob', place='Paris')}\n")

print("3. Fetching by commit_id...")
fetched = client.prompts.get(
    project_name="test-project",
    name="greeting",
    commit_id=prompt_v1.commit_id,
)
assert fetched is not None
print(f"   Fetched v1: {fetched.prompt[:50]}...")
print(f"   Metadata: {fetched.metadata}\n")

print("4. Fetching by tag...")
fetched_by_tag = client.prompts.get(
    project_name="test-project",
    name="greeting",
    tag="latest",
)
assert fetched_by_tag is not None
print(f"   Fetched by 'latest' tag: {fetched_by_tag.prompt[:50]}...")
print(f"   Tags: {fetched_by_tag.tags}\n")

print("5. Adding additional tag to v1...")
new_commit_id = client.prompts.tag(
    project_name="test-project",
    name="greeting",
    commit_id=prompt_v1.commit_id,
    tags=["production"],
)
print(f"   Tagged commit: {new_commit_id}\n")

print("6. Listing all versions...")
versions = client.prompts.list(
    project_name="test-project",
    name="greeting",
)
print(f"   Found {len(versions)} versions:")
for i, v in enumerate(versions, 1):
    print(f"   {i}. commit={v.commit_id[:8]}, tags={v.tags}, created={v.created_at}")
print()

print("7. Removing tag...")
removed_commits = client.prompts.untag(
    project_name="test-project",
    name="greeting",
    tags=["v2.0"],
)
print(f"   Removed 'v2.0' tag from {len(removed_commits)} commit(s)\n")

print("8. Testing compile with missing variable...")
try:
    prompt_v1.compile(name="Charlie")
except ValueError as e:
    print(f"   Expected error: {e}\n")

print("=== All tests completed ===")

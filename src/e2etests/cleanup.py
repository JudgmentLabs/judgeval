import os

from e2etests.utils import delete_project


def main() -> None:
    registry_path = os.environ["E2E_CREATED_PROJECTS_FILE"]
    if not os.path.exists(registry_path):
        return
    with open(registry_path) as registry:
        project_names = sorted({line.strip() for line in registry if line.strip()})
    failures = []
    for project_name in project_names:
        try:
            delete_project(project_name=project_name)
        except Exception as error:
            failures.append(f"{project_name}: {error}")
    print(
        f"Cleaned up {len(project_names) - len(failures)}/{len(project_names)} e2e projects"
    )
    for failure in failures:
        print(f"Failed to delete {failure}")


if __name__ == "__main__":
    main()

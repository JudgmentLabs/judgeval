import requests
from dotenv import load_dotenv
import os


def test():
    response = requests.post(
        "http://localhost:8000/organization/fetch_all_organizations_for_user/",
        json={},
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('JUDGMENT_API_KEY')}",
        },
    )
    print(response.json())


def test2():
    response = requests.post(
        "http://localhost:8000/organization/create/",
        json={"organization_name": "test_organization2"},
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('JUDGMENT_API_KEY')}",
        },
    )
    print(response.json())


def main():
    load_dotenv()
    test()


if __name__ == "__main__":
    main()

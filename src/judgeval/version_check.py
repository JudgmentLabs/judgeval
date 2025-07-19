import importlib.metadata
from judgeval.utils.requests import requests
import threading
from judgeval.common.logger import judgeval_logger


def check_latest_version(package_name: str = "judgeval"):
    def _check():
        try:
            current_version = importlib.metadata.version(package_name)
            response = requests.get(
                f"https://pypi.org/pypi/{package_name}/json", timeout=2
            )
            latest_version = response.json()["info"]["version"]

            if current_version != latest_version:
                judgeval_logger.warning(
                    f"UPDATE AVAILABLE: You are using '{package_name}=={current_version}', "
                    f"but the latest version is '{latest_version}'. While this version is still supported, "
                    f"we recommend upgrading to avoid potential issues or missing features: "
                    f"`pip install --upgrade {package_name}`"
                )
        except Exception:
            pass

    threading.Thread(target=_check, daemon=True).start()

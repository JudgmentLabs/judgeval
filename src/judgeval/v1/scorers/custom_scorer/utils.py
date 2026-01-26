import importlib.util
from judgeval.logger import judgeval_logger
from pathlib import Path


def extract_scorer_name(scorer_file_path: str) -> str:
    try:
        spec = importlib.util.spec_from_file_location("scorer_module", scorer_file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec from {scorer_file_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and any("Scorer" in str(base) for base in attr.__mro__)
                and attr.__module__ == "scorer_module"
            ):
                try:
                    scorer_instance = attr()
                    if hasattr(scorer_instance, "name"):
                        return scorer_instance.name
                except Exception:
                    continue

        raise AttributeError("No scorer class found or could be instantiated")
    except Exception as e:
        judgeval_logger.warning(f"Could not extract scorer name: {e}")
        return Path(scorer_file_path).stem

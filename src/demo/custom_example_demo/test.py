from judgeval.data import CustomExample
from judgeval import JudgmentClient
from qodo_example import QodoExample

custom_example = CustomExample(
    code="print('Hello, world!')",
    original_code="print('Hello, world!')",
)

qodo_example = QodoExample(
    code="print('Hello, world!')",
    original_code="print('Hello, world!')",
)

print(f"{custom_example=}")
print(f"{qodo_example=}")
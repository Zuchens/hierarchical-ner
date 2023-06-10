import typing
from dataclasses import dataclass
from typing import Any


@dataclass
class RawDocument:
    offsets2Entities: dict[str, Any]  # pylint: disable=invalid-name
    dependencies: list[list[str]]
    dependencyLabels: list[list[str]]  # pylint: disable=invalid-name
    entities: list[list[dict[str, Any]]]
    tokens: list[str]
    text: str

    @property
    def offsets(self) -> list[int]:
        return sorted([int(token) for token in list(self.offsets2Entities.keys())])

    @property
    def dependencies_with_labels(self) -> typing.Iterator:
        return zip(self.dependencies, self.dependencyLabels)

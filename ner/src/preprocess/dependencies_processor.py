from dataclasses import dataclass


@dataclass
class Dependency:
    dependency_idx: str
    dependency_label: str
    dependency_label_idx: int


class DependenciesProcessor:

    def __init__(self):
        self.dependency_label_to_idx = {}
        self.dependency_label_iterator = 0

    def set_dependencies(self, dependencies_idx: list[str], dependencies_label: list[str]) -> list[Dependency]:
        dependencies = []
        for dependency_idx, dependency_label in zip(dependencies_idx, dependencies_label):
            if dependency_label not in self.dependency_label_to_idx:
                self.dependency_label_to_idx[dependency_label] = self.dependency_label_iterator
                self.dependency_label_iterator += 1
            if dependency_idx == "None":
                dependency_idx = "-2"
            dependencies.append(
                Dependency(dependency_idx=dependency_idx,
                           dependency_label=dependency_label,
                           dependency_label_idx=self.dependency_label_to_idx[dependency_label]))
        return dependencies

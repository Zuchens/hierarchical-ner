from unittest import TestCase

from ner.src.preprocess.dependencies_processor import Dependency, DependenciesProcessor


class DependenciesProcessorTestCase(TestCase):

    def test_set_dependencies(self):
        # GIVEN
        dependencies_idx = ["-1", "3", "2", "0"]
        dependencies_labels = ["root", "obj", "obj", "punct"]
        expected_dependencies = [
            Dependency(dependency_idx="-1", dependency_label="root", dependency_label_idx=0),
            Dependency(dependency_idx="3", dependency_label="obj", dependency_label_idx=1),
            Dependency(dependency_idx="2", dependency_label="obj", dependency_label_idx=1),
            Dependency(dependency_idx="0", dependency_label="punct", dependency_label_idx=2),
        ]
        expected_label_to_idx = {"root": 0, "obj": 1, "punct": 2}
        dependencies_processor = DependenciesProcessor()

        # WHEN
        result = dependencies_processor.set_dependencies(dependencies_idx=dependencies_idx,
                                                         dependencies_label=dependencies_labels)

        # THEN
        self.assertListEqual(expected_dependencies, result)
        self.assertDictEqual(expected_label_to_idx, dependencies_processor.dependency_label_to_idx)

    def test_set_dependencies_with_none(self):
        # GIVEN
        dependencies_idx = ["-1", "3", "None", "0"]
        dependencies_labels = ["root", "obj", "obj", "punct"]
        expected_dependencies = [
            Dependency(dependency_idx="-1", dependency_label="root", dependency_label_idx=0),
            Dependency(dependency_idx="3", dependency_label="obj", dependency_label_idx=1),
            Dependency(dependency_idx="-2", dependency_label="obj", dependency_label_idx=1),
            Dependency(dependency_idx="0", dependency_label="punct", dependency_label_idx=2),
        ]
        expected_label_to_idx = {"root": 0, "obj": 1, "punct": 2}
        dependencies_processor = DependenciesProcessor()

        # WHEN
        result = dependencies_processor.set_dependencies(dependencies_idx=dependencies_idx,
                                                         dependencies_label=dependencies_labels)

        # THEN
        self.assertListEqual(expected_dependencies, result)
        self.assertDictEqual(expected_label_to_idx, dependencies_processor.dependency_label_to_idx)

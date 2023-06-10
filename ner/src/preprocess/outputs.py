from collections import defaultdict
from typing import Any

from ner.src.common.constants import Constants
from ner.src.common.preprocessed.document import Document

RawTargetType = list[dict[str, Any]]
LabeledTargetType = set[str]


class OutputProcessor:

    def __init__(self, labels):
        self.labels = labels or {Constants.pad_label: 0}
        self.label_to_idx_iterator = len(self.labels)

    def get_targets_idx(self, documents: list[Document]) -> list[list[int]]:
        targets = [document.raw_document.entities for document in documents]
        target_labels = self.get_target_label_lists(targets)
        target_idx = self.convert_concatenated_labels_to_indices(target_labels)
        return target_idx

    def convert_concatenated_labels_to_indices(self, targets: list[list[set[str]]]) -> list[list[int]]:
        targets_idx: list[list[int]] = []
        target_to_idx = defaultdict(int)
        for idx, sentence_targets in enumerate(targets):
            sentence_target_idx: list[int] = []
            for word_target in sentence_targets:
                word_target = sorted(set(word_target))
                concatenated_label = "-".join(word_target)
                if concatenated_label not in self.labels:
                    self.labels[concatenated_label] = self.label_to_idx_iterator
                    self.label_to_idx_iterator += 1
                target_to_idx[concatenated_label] += 1
                sentence_target_idx.append(self.labels[concatenated_label])
            targets_idx.append(sentence_target_idx)
        return targets_idx

    def get_target_label_lists(self, document_targets: list[list[RawTargetType]]) -> list[list[LabeledTargetType]]:
        return [self.get_sentence_target_labels(sentence_targets) for sentence_targets in document_targets]

    def get_sentence_target_labels(self, raw_sentence_targets: list[RawTargetType]) -> list[LabeledTargetType]:
        sentence_target_labels: list[set[str]] = [set() for _ in range(len(raw_sentence_targets))]
        for idx, targets in enumerate(raw_sentence_targets):
            if len(targets) > 0:
                for target in targets:
                    ne_type = f"{target['type']}_{target['subtype']}" if target.get("subtype") else target["type"]
                    # handle targets that span through multiple words
                    for i in range(len(target["offsets"])):
                        if len(sentence_target_labels) >= (idx + i + 1):
                            sentence_target_labels[idx + i].add(ne_type)
            else:
                if not sentence_target_labels[idx]:
                    sentence_target_labels[idx].add(Constants.outside_label)
        return sentence_target_labels

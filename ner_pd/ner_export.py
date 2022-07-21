import json
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ner import SpacyTransitionModel, SpacyTransformer, DslimNER

"""
Source file used for being compliant with Cisco
"""


@dataclass
class NamedEntity:
    type: str
    text: str


@dataclass
class EmailResult:
    mail_id: str
    named_entities: list[list[NamedEntity]]


class NERExporter(ABC):
    @abstractmethod
    def export_named_entities(self, sentences: list[str]) -> list[list[NamedEntity]]:
        pass

    def process(self, mail_id: str, sentences: list[str]) -> EmailResult:
        return EmailResult(
            mail_id=mail_id, named_entities=self.export_named_entities(sentences)
        )


class DummyNERExporter(NERExporter):
    def __init__(self):
        # identifier = "en_core_web_sm" # or its sm/md variant
        # self.model = SpacyTransition(identifier)

        # identifier = "dslim/bert-base-NER" #"dslim/bert-large-NER"
        # self.model = DslimNER(identifier)

        identifier = "en_core_web_trf"
        self.model = SpacyTransformer(identifier)

    def get_sentence_records(self, sentence: str) -> list[NamedEntity]:
        self.model.run_on_sentence(sentence)
        entities = self.model.get_entities()
        sentence_records = []
        for ent in entities:
            sentence_records.append(ent)
        return sentence_records

    def export_named_entities(self, sentences: list[str]) -> list[list[NamedEntity]]:
        all_records = []
        for sentence in sentences:
            records = self.get_sentence_records(sentence)
            all_records.append(records)
        return all_records


if __name__ == "__main__":
    ner_exporter = DummyNERExporter()
    emails = {
        "mail_1": ["Pay me 1000 CZK.", "Nothing here.", "We all live in Prague."],
        "mail_2": [
            "I still want that 1000 CZK.",
            "Still nothing.",
            "Prague is my favourite city.",
        ],
        "examples_from_sents/sentences.csv": [
            "Treasurer - Enron Corp Ray Bowen was elected to Executive Vice President, Finance and Treasurer of Enron Corp recently.",
            "I have advised Mark that we will not begin the drafting process until we have heard back from him that the terms are acceptable.",
            "We apologize for any inconvenience this may cause.",
            "Are the rules different in Australia?",
        ],
    }
    for mail_id, sentences in emails.items():
        result = ner_exporter.process(mail_id, sentences)
        print(json.dumps(result, default=vars))

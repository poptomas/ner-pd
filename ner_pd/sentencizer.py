import spacy
import random
import re


class Sentencizer:
    """
    Wrapped spaCy Sentencizer with
    sentence cleaning mechanisms
    detached from the rest of the files
    """

    def __init__(self, text):
        self._document = self.__load_sentencizer(text)
        self._seed = 0
        self._sentences = []
        self.filter_sentences()

    def filter_sentences(self):
        """
        Basic sentence cleaning by trimming,
        in case the sentence is empty
        afterward, it is not taken into consideration
        """
        for sent in self._document.sents:
            sentence = sent.text.strip()
            if sentence:
                self._sentences.append(sentence)

    def get_random_sentence(self):
        """
        Returns a pseudorandom sentence from parsed sentences
        """
        random.seed(self._seed)
        index = random.randint(0, len(self._sentences) - 1)
        self._seed = index
        return self.__get_cleaned_sentence(self._sentences[index])

    def get_sentences(self) -> list[str]:
        return self._sentences

    def __load_sentencizer(self, text: str):
        """
        Load rule-based sentence parsing option Sentencizer component included in the spaCy library
        - https://spacy.io/api/sentencizer
        """
        reduced_model = spacy.blank("en")
        reduced_model.add_pipe("sentencizer")
        return reduced_model(text)

    def __load_senter(self, text: str):
        """
        Load Senter (SentenceRecognizer) component included in the spaCy library
        -  https://spacy.io/api/sentencerecognizer
        - note - a more precise sentence parsing variant, however, more computationally expensive than Sentencizer
        - eventually unused
        """
        reduced_model = spacy.load(
            "en_core_web_sm",
            disable=[
                "tok2vec",
                "tagger",
                "parser",
                "attribute_ruler",
                "lemmatizer",
                "ner",
            ],
        )
        reduced_model.enable_pipe("senter")
        return reduced_model(text)

    def __get_cleaned_sentence(self, input_sentence: str) -> str:
        """
        Sentence cleaning by trimming (and ignoring the empty entries)
        and reducing mutliple spaces within the sentence
        """
        input_sentence = input_sentence.strip()
        cleaned = re.sub("\s+", " ", input_sentence)
        return cleaned

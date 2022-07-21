import pandas as pd
from tqdm.auto import tqdm

import argparse
import random
import email

from .sentencizer import Sentencizer
from .utils import DatasetUtils


class EmailProcessor:
    """
    Email creation class
    - processes a pseudorandomly
    chosen row from the provided csv dataset,
    and, using email Python package,
    parser an email to a string form for further use
    """

    def __init__(self, args: argparse.Namespace):
        self._filename = args.filename
        self._index = 0
        self._text = ""
        self._debug = False  # may be (and also was) useful for a concrete email lookup
        self._seed = 0
        self.__receive_dataset()

    def __receive_dataset(self):
        """
        Utilizes pandas to read the dataset in the comma separated values format
        """
        data = pd.read_csv(self._filename, engine="c", iterator=True, chunksize=1024)
        self.dataset = pd.concat(data, ignore_index=True)
        self._length = len(self.dataset)

    def create_email(self):
        """
        Pseudorandomly chooses an email from the dataset,
        parses it to a string form
        """
        col_name = "message"
        messages = self.dataset[col_name]
        random.seed(self._seed)
        self._index = random.randint(0, self._length - 1)
        self._seed = self._index
        if self._debug:
            print("[Index {}] (pseudo)randomly chosen".format(self._index))
        example = messages[self._index]
        self.__form_text(example)

    def __form_text(self, example: str):
        """
        Parses email using "email" package https://docs.python.org/3/library/email.html
        and obtains the email content in the string form
        """
        message = self.__parse_email(example)
        self._text = message.get_payload()

    def __parse_email(self, input_text: str) -> email.message.Message:
        """
        Returns email message input text representation from string representation
        """
        return email.parser.Parser().parsestr(input_text)

    def get_text(self) -> str:
        return self._text


class SentencesProcessor(EmailProcessor):
    """
    Sentence processor utilizes the csv email dataset collection
    to form own dataset which was annotated by the author
    in advance (evaluations can be found in the "gold" directory)
    """

    def __init__(self, args: argparse.Namespace, outfile: str):
        self.table_fname = outfile
        super().__init__(args)

    def build_dataset(self, num_records: int):
        """
        Form dataset from pseudorandomly sampled emails,
        and obtain a random sentence from the email
        """
        self.__overwrite()
        for _ in tqdm(  # visual
            range(num_records), colour="red", desc="Getting sentences"
        ):
            self.__append()

    def __get_sentence(self) -> list[str]:
        """
        Utilizes sentencizer to parse email into sentences,
        returns pseudo-random one
        """
        self.create_email()
        sent = Sentencizer(self._text)
        sentence = sent.get_random_sentence()
        row_records = [sentence]
        return row_records

    def __append(self):
        """
        Append a pseudo-random sentence from an email to the dataset
        """
        row_records = self.__get_sentence()
        DatasetUtils.append_row_to_csv(
            self.table_fname, row_records=row_records, col_names=None
        )

    def __overwrite(self):
        """
        Flush the last run
        """
        with open(self.table_fname, "w"):
            pass

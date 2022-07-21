from dataclasses import dataclass
import io
from typing import Tuple
import pandas as pd
import difflib
import time
import argparse
import os
import shutil


def benchmark(func):
    def function_timer(*args, **kwargs):
        start = time.perf_counter()
        value = func(*args, **kwargs)
        end = time.perf_counter()
        runtime = end - start
        round_time = round(runtime * 1000, 2)
        message = "[{}] took {} ms".format(func.__name__, round_time)
        print(message)
        FormatUtils.printline()
        return value

    return function_timer


class FormatUtils:
    @staticmethod
    def printline():
        line_length = 30
        print("―" * line_length)

    @staticmethod
    def print_arguments(args: argparse.Namespace):
        FormatUtils.printline()
        print("Running with arguments:")
        for arg in vars(args):
            print("{}: {}".format(arg, getattr(args, arg)))
        FormatUtils.printline()

    @staticmethod
    def print_error_with_dataset(filename: str):
        print('A problem with "{}" dataset occurred.'.format(filename))

    @staticmethod
    def print_identifier(identifier: str):
        print("Used model: {}".format(identifier))


class StringBuilder:
    def __init__(self):
        self._string = io.StringIO()
        self._length = 0

    def append(self, *arguments):
        for member in arguments:
            word_len = self._string.write(member)
            self._length += word_len

    def to_string(self) -> str:
        return self._string.getvalue()

    def clear(self):
        self._string = io.StringIO()

    def is_empty(self) -> bool:
        return self._length == 0


class StringUtils:
    @staticmethod
    def get_string_from_set(in_set: set):
        if len(in_set) == 0:
            return "---"
        else:
            return " ".join(in_set)

    @staticmethod
    def has_same_prefix(first: str, second: str, count: int) -> bool:
        return first[0:count] == second[0:count]

    @staticmethod
    def trim_front(input_v: str, index: int) -> str:
        return input_v[index:]

    @staticmethod
    def __matches_with_relaxed_sequence_matching(
        large_string: str, query_string: str, threshold=0.85
    ):
        # Exact match was way too strict
        # (i. e. Creditors' Committee vs. Creditors'Committee found by bert-base-NER was declined)
        matcher = difflib.SequenceMatcher(None, large_string, query_string)
        match = "".join(
            large_string[idx : idx + match_]
            for idx, _, match_ in matcher.get_matching_blocks()
            if match_
        )
        return len(match) / float(len(query_string)) >= threshold

    @staticmethod
    def trim_front_if_contained(large_string: str, query_string: str) -> str:
        if (
            query_string in large_string
            or StringUtils.__matches_with_relaxed_sequence_matching(  # a viable alternative for i.e. bert-base-NER
                large_string, query_string
            )
        ):
            return StringUtils.trim_front(large_string, len(query_string))
        else:
            return large_string

    @staticmethod
    def format_found_ents(found_ents: list) -> str:
        empty_record_sign = "---"
        builder = StringBuilder()
        for mem in found_ents:
            builder.append("[{}: {}]".format(mem.type, mem.text))
            if mem != found_ents[-1]:
                builder.append(" ")
        result = builder.to_string()
        if not result:
            result = empty_record_sign
        return result


class DatasetUtils:
    @staticmethod
    def truncate_dataset(fname: str, end_index: int):
        try:
            dataframe = pd.read_csv(fname)
            dataframe = dataframe[0:end_index]
            dataframe.to_csv(fname, index=False)
        except:
            print("File [{}] not found".format(fname))

    @staticmethod
    def write_row_to_csv(
        table_fname: str, row_records: list, col_names: list, header: bool
    ):
        dataframe = pd.DataFrame(data=row_records, columns=col_names, dtype=str)
        dataframe.to_csv(table_fname, index=False, header=header)

    @staticmethod
    def append_row_to_csv(table_fname: str, row_records: list, col_names: list):
        dataframe = pd.DataFrame(data=row_records, columns=col_names, dtype=str)
        dataframe.to_csv(table_fname, index=False, mode="a", header=False)


@dataclass
class Stat:
    outcome: str
    count: int


class DictionaryUtils:
    class DictionaryWrapper(dict):
        def __missing__(self, key):
            value = self[key] = type(self)()
            return value

    @staticmethod
    def increment_occurrence(dictionary: dict, key: str):
        if key in dictionary:
            dictionary[key] += 1
        else:
            dictionary[key] = 1

    @staticmethod
    def increment_nested_occurrence(dictionary: dict, first_key: str, second_key: str):
        if first_key in dictionary:
            found = False
            for v in dictionary[first_key]:
                if v.outcome == second_key:
                    v.count += 1
                    found = True
                    break
            if not found:
                dictionary[first_key].append(Stat(outcome=second_key, count=1))
        else:
            dictionary[first_key] = [Stat(outcome=second_key, count=1)]


class FilesystemUtils:
    @staticmethod
    def clean_previous_run(directory: str):
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)

    @staticmethod
    def clean_up_besides(out_formats, directory: str):
        files_in_directory = os.listdir(directory)
        filtered_files = [
            file for file in files_in_directory if not file.endswith(tuple(out_formats))
        ]
        for file in filtered_files:
            path_to_file = os.path.join(directory, file)
            os.remove(path_to_file)

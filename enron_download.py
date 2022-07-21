import requests
import shutil
import os
import pandas as pd
from tqdm.auto import tqdm
from ner_pd.utils import benchmark


class Enron:
    """
    Enron email dataset downloader + serializer compliant with the
    format published on https://www.kaggle.com/datasets/wcukierski/enron-email-dataset
    - in case the user did not want to run the script - python enron.py
    """

    def __init__(self, tar_fname: str, csv_fname: str, ds_fname: str):
        self.tar_fname = tar_fname
        self.csv_fname = csv_fname
        self.dataset_dir = ds_fname

    @benchmark
    def download_dataset(self, url: str, chunk_size: int):
        """
        Downloads the Enron email dataset using requests library while displaying the progress bar
        """
        request = requests.get(url, stream=True)
        size = float(request.headers["content-length"])
        progress_bar = tqdm(total=size, unit="iB", unit_scale=True, colour="red")
        with open(self.tar_fname, "wb") as file:
            # in order to prevent loading everything in the memory at once
            for chunk in request.iter_content(chunk_size=chunk_size):
                progress_bar.update(len(chunk))
                file.write(chunk)

    @benchmark
    def unzip_dataset(self):
        shutil.unpack_archive(self.tar_fname, ".")

    @benchmark
    def build_csv(self):
        """
        Form the csv file compliant with the kaggle Enron dataset
        - sorted order required
        """
        for root, _, files in sorted(os.walk(self.dataset_dir)):
            for fname in sorted(files):
                path = os.path.join(root, fname)
                self.__add_record(path)

    @benchmark
    def clean_up(self):
        """
        Enron unzipping and processing leaves behind large files
        from which the processing was made - no longer useful
        """
        os.remove(self.tar_fname)
        shutil.rmtree(self.dataset_dir)

    def __add_record(self, path: str):
        """
        add a row to a csv file with the structure:
        file       | message
        identifier | email contents including header
        """
        prefix = self.dataset_dir
        fname = path.lstrip(prefix)[1:]
        with open(path, "r", encoding="cp1252") as file:
            content = "".join(file.readlines())
            row = [{"file": fname, "message": content}]
            dataframe = pd.DataFrame(row)
            if os.path.isfile(self.csv_fname):
                dataframe.to_csv(self.csv_fname, mode="a", index=False, header=False)
            else:
                dataframe.to_csv(self.csv_fname, index=False, header=True)


if __name__ == "__main__":
    # last revised version of the Enron email dataset from 2015
    url = "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"
    tar_gz_fname = "enron.tar.gz"
    csv_fname = "enron.csv"
    dataset_dir = "maildir"
    enron = Enron(tar_gz_fname, csv_fname, dataset_dir)
    enron.download_dataset(url, chunk_size=1024)

    enron.unzip_dataset()
    enron.build_csv()
    enron.clean_up()

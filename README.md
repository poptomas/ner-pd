
#  Named Entity Recognition and Its Application to Phishing Detection

##  Table of Contents

- [Named Entity Recognition and Its Application to Phishing Detection](#named-entity-recognition-and-its-application-to-phishing-detection)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Technology Used](#technology-used)
    - [Python](#python)
    - [External dependencies](#external-dependencies)
      - [Pandas](#pandas)
      - [PyTorch](#pytorch)
      - [tqdm](#tqdm)
      - [transformers](#transformers)
      - [spaCy](#spacy)
      - [NumPy](#numpy)
      - [flair](#flair)
      - [spaCy models](#spacy-models)
  - [Setup](#setup)
    - [1) Download Python 3.9+](#1-download-python-39)
      - [Linux](#linux)
        - [Windows](#windows)
    - [2) Virtual Environment](#2-virtual-environment)
      - [Linux](#linux-1)
      - [Windows](#windows-1)
  - [Dataset Preparation](#dataset-preparation)
    - [Github](#github)
    - [Non-Github](#non-github)
  - [Launch](#launch)
    - [Enron Experiment](#enron-experiment)
    - [Annotation Experiment](#annotation-experiment)
    - [Benchmark Experiment](#benchmark-experiment)
    - [Divergence Experiment](#divergence-experiment)
    - [ROC Experiment](#roc-experiment)
  - [Hardware Requirements](#hardware-requirements)

##  Introduction

This project comprises source files that served as support material while writing the bachelor thesis
with the primary aim to utilize such findings in the thesis. Eventually, the subprograms utilized
were summarized into multiple entry point sources denoted by ```*_experiment.py``` 
from which the experiments conducted can be replicated. The functionality covered by this library is further described in
[Launch](#launch) and particular experiments in its subsections.

##  Technology Used

### Python 
- Python 3.9+ is required due to utilized language constructs
- Nevertheless, be aware that with Python 3.10, the ```flair``` library would show deprecation warnings (which were suppressed for this reason)

### External dependencies
Various external Python libraries were used, and there are versions used for the ideal use (via the virtual environment).

#### Pandas
- data manipulation and analysis library
- Pandas library was utilized for CSV handling (dataset loading, dataset building, CSV as a simple serialization format)
- version 1.4.1

#### PyTorch
- open source machine learning framework mainly for deep learning models development
- dependency required for software for NER to be able to utilize GPU computation
- version 1.11.0

#### tqdm
- since the project is a command line collection of conducted experiments, 
and for most of them take quite some time to obtain feedback, the progress bar comes in useful
- version 4.64.0

#### transformers
- transformers library is required for BERT_base, BERT_large, NER models from Hugging Face
- version 4.17.0

#### spaCy
- NLP multipurpose library, here used as software for NER
- version 3.2.4

#### NumPy
- powerful library for effective array/matrix/tensor manipulation, comprehensive mathematical functions, random number generators
- version 1.22.3

#### flair
- NER + Part-of-speech library, here used as software for NER
- BiLSTM, RoBERTa-large models both for CoNLL 2003 and OntoNotes v5
- version 0.11.3 
  - be warned that although the library is the latest release (20/05/2022), 
  Python 3.10 would display deprecation warnings due to obsolete huggingface-hub cached downloadS

#### spaCy models
- en_core_web_sm, en_core_web_md, en_core_web_lg, en_core_web_trf were utilized
- en_core_web_{sm, md, lg} - transition-based models
  - [en_core_web_sm](https://spacy.io/models/en#en_core_web_sm)
  - [en_core_web_md](https://spacy.io/models/en#en_core_web_md)
  - [en_core_web_lg](https://spacy.io/models/en#en_core_web_lg) 
- en_core_web_trf - transformer-based model - RoBERTa-base
  - [en_core_web_trf](https://spacy.io/models/en#en_core_web_trf)
- version 3.2.0

##  Setup

### 1) Download Python 3.9+

#### Linux 
The procedure is demonstrated on Ubuntu (the author used WSL Ubuntu 22.04)
```
sudo apt -y update
sudo apt -y install software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt -y install python3.9
sudo apt -y install python3.9-venv
alias python=python3.9  # (optional) to simplify the scripts, otherwise, for instance, python3.9 enron_experiment.py [args] is required
```
- Taken from https://linuxize.com/post/how-to-install-python-3-9-on-ubuntu-20-04/

##### Windows 
- For instance, Python 3.9 can be downloaded from https://www.python.org/downloads/release/python-399/
- venv comes with the installer

### 2) Virtual Environment

- highly recommended
- to avoid libraries version conflicts
- Note that all scripts shown use the "aliased" version of Python (python3.* instead should work out too)
- All concrete versions with libraries needed are obtained using the commands below (in terminal):

#### Linux
```
python -m venv venv
source venv/bin/activate
pip install -e .
```

#### Windows
```
python -m venv venv
venv\Scripts\activate
pip install -e .
```

## Dataset Preparation

### Github 

```enron.csv``` was uploaded via git lfs to the git repository.

### Non-Github

Enron experiment and annotation experiment rely on the Enron Email Dataset
which needs to be downloaded, unzipped, and serialized into the CSV format 
with the compliant structure of an alternative published on [Kaggle](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset):

```
python enron_download.py
```

Alternatively, the Enron email dataset can be downloaded directly from [Kaggle](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset)
and moved to the root directory of the project. It was tested that both approaches are interchangeable for the experiments.
For the other experiments (benchmark, divergence, and ROC), example data
are provided (contained in the ```data``` directory) to run the experiments with them.

##  Launch

The project contains various entrypoint "source" files in the root directory
based on the experiment which is supposed to be launched:
- [enron_experiment.py](#enron-experiment)
- [annotation_experiment.py](#annotation-experiment)
- [benchmark_experiment.py](#benchmark-experiment)
- [divergence_experiment.py](#divergence-experiment)
- [roc_experiment.py](#roc-experiment)

### Enron Experiment

Enron experiment utilizes various variants of models for named entity recognition while processing Enron emails.
As an output, it produces 
  - csv pairwise files of comparisons in case a group/full mode launch was utilized 
    - mainly used for spacy models to find anomalies in named entity differences found between transformer and transition-based models
  - json produced contains named entity occurrences - used for the KL/JS divergence experiment
  - csv containing per-model count of each named entity type found 

A command to reproduce the experiment:
```
python enron_experiment.py
```

- defaults set to
  - ```--filename``` being ```enron.csv``` - assuming [Dataset Preparation](#dataset-preparation) 
  part using the provided ```enron_download.py``` script was conducted, otherwise, for the [Kaggle](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset) downloaded dataset, change ```--filename=emails.csv``` assuming the dataset remains in the root directory
  - to be reasonably fast for demonstration purposes
    - ```--email_count=100```
    - ```--mode=md```
    - ```--outdir=out```
    - results in ```python enron_experiment.py --email_count=100 --mode=md --outdir=out```
- ```--email_count``` - the number of emails - up to the capacity of the dataset, in case of the Enron email dataset 517401 (not recommended though)
- ```--mode```
  - Various modes are available 
    - NER library group based: 
      - pairwise comparisons between models regarding named entities found are conducted
      - --mode=```spacy``` - launches ```sm```, ```md```, ```lg```, ```trf``` 
      - --mode=```huggingface``` - launches ```hbase```, ```hlarge```
      - --mode=```flair``` - launches ```fltrf```, ```fllstm```
    - concrete models: 
      - --mode=```sm```, --mode=```md```, --mode=```lg``` - transition-based spaCy models differing in the size of word vectors used for pre-training
      - --mode=```trf``` - spaCy - [RoBERTa-base](https://spacy.io/models/en#en_core_web_trf)
      - --mode=```hbase``` - Hugging Face - [dslim-base-NER](https://huggingface.co/dslim/bert-base-NER)
      - --mode=```hlarge``` - Hugging Face - [dslim-large-NER](https://huggingface.co/dslim/bert-large-NER)
      - --mode=```fltrf``` - Flair - [RoBERTa-large](https://huggingface.co/flair/ner-english-ontonotes-large)
      - --mode=```fllstm``` - Flair - [BiLSTM](https://huggingface.co/flair/ner-english-ontonotes-fast)
- ```--outdir``` - directory where to store the results of the experiment

### Annotation Experiment

Annotation experiment utilizes hand-made annotations on 200 sentences from the Enron Email Dataset. The sentences are formed as a dataset first, then all models fine-tuned on OntoNotes v5 (spaCy, flair) are launched and requested to do named entity recognition task on this dataset.

Per-entity precision, recall, F1-score, and relaxed F1-score are calculated. For each entity, a CSV file
is produced with the model's predictive performance comparison, mainly used for a thorough analysis between models. 

Each model produces its CSV table where named entities found
are divided into four categories - True positive, false positive, false negative, and almost true positive (which is used for forming a relaxed F1-score instead of true positive) for a thorough analysis of how each model performed (the author's gold data are contained in ```gold``` directory).

The most important part of the output produces ```score.csv``` where precision, recall, F1-score, relaxed F1-score is computed for each model forming a CSV table, as shown in the thesis.

A command to reproduce the experiment:
```
python annotation_experiment.py --mode=spacy
```

- defaults set to 
    - ```--filename```- by default ```--filename=enron.csv``` - assuming [Dataset Preparation](#dataset-preparation) part using the author's provided script was conducted, otherwise, for the Kaggle downloaded dataset, change it to ```--filename=emails.csv``` assuming the dataset remains in the root directory
    - ```--outdir=sents``` by default
    -```--mode=spacy``` by default to show the predictive performacne comparison
- ```--filename```
- ```--outdir``` - directory where to store the results of the experiment
- ```--mode``` - same as in the [Enron experiment](#enron-experiment)

### Benchmark Experiment

Benchmark experiment measures the speed of NER models on 10 000 sentences prepared beforehand by the author (```data/seed{0,16,32}_10k.csv```)

A command to reproduce the experiment:
```
python benchmark_experiment.py
```
- defaults set to run on 10 000 sentences (data/seed0_10k.csv) with the best speed:F1-score ratio (otherwise, according to conducted
measurments, the best predictive performance is available using spaCy transformer - en_core_web_trf)

- ```--mode``` - the choice of the NER model 
  - sm, md, lg - transition-based models by spaCy
  - trf - transformer-based (RoBERTa-base) model by spaCy
  - hbase - transformer BERT-base model from Hugging Face
  - hlarge - transformer BERT-large model from Hugging Face
  - fllstm - flair BiLSTM model
  - fllstmdefault - flair BiLSTM model (slower)
  - fltrf - flair transformer-based model (RoBERTa-large)

- ```--sentences``` - number of sentences taken into consideration  - defaults to 10 000 but a smaller size of the dataset can be used (or in case custom sentence dataset with more than 10k sentences is formed, more sentences can be taken into consideration)
-```--dir``` - the directory from which the sentence dataset is taken from (by default ```data```, do not change unless own dataset is utilized)
-```--filename``` - filename - from prepared datasets, ```seed0_10k.csv```, ```seed_16_10k.csv```, or ```seed_32_10k.csv``` can be utilized
- a quick demonstration example with all filled arguments: ```python benchmark_experiment.py --mode=md --sentences=100 --dir=data --filename=seed0_10k.csv```

### Divergence Experiment

Divergence experiment visualizes named entity probability distributions. Based on the probability distributions, Jensen-Shannon divergence
is computed and visualized. The visualization consists of probability distributions (alltogether + pairwise) and JS divergence (alltogether + pairwise).

A command to reproduce the experiment:
```
python divergence_experiment.py --occurrences
```

- ```--occurrences``` flag is used whether the whole named entity probability distribution is supposed to be taken into consideration for the Jensen-Shannon divergence computation or whether only its (non)existence in a sentence should be considered 

- default for the output directory is set to directory ```stats_out``` but optionally ```--outdir``` can be used to specify the directory where to store the generated plots

### ROC Experiment
ROC (Receiver Operating Characteristic) curves are visualized based on provided JSON data from ```data/NER-Results3-ExportFloat32.json```
which contain results of the phishing email classifier experiment - whether named entities enrichment helps predictive performance

A command to reproduce the experiment:
```
python roc_experiment.py
```

- default for the output directory is set to directory ```stats_out``` but optionally ```--outdir``` can be used to specify the directory where to store the generated plots

##  Hardware Requirements

To run the experiments with GPU, an Nvidia GPU is required to utilize CUDA (PyTorch requires => transformer-based models require), nevertheless,
transformer-based models can run on CPU too (with a warning that GPU is not provided for the computation).

The experiments were conducted on a device with AMD Ryzen 7 4800H, 16GB RAM, Nvidia RTX 2060 6GB, OS Windows using WSL Ubuntu 22.04.

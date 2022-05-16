"""Arabic Speech Corpus"""

from __future__ import absolute_import, division, print_function

import os

import datasets
import soundfile as sf

_CITATION = """
"""

_DESCRIPTION = """\


```python
import soundfile as sf

def map_to_array(batch):
    speech_array, _ = sf.read(batch["file"])
    batch["speech"] = speech_array
    return batch

dataset = dataset.map(map_to_array, remove_columns=["file"])
```
"""

_URL = "mgb3.zip"
corrupt_files = [
    "familyKids_02_first_12min.wav",
    "sports_04_first_12min.wav",
    "cooking_05_first_12min.wav",
    "moviesDrama_07_first_12min.wav",
    "science_06_first_12min.wav",
    "comedy_09_first_12min.wav",
    "cultural_08_first_12min.wav",
    "familyKids_11_first_12min.wav",
    "science_10_first_12min.wav",
]


class EgyptianSpeechCorpusConfig(datasets.BuilderConfig):
    """BuilderConfig for EgyptianSpeechCorpus."""

    def __init__(self, **kwargs):
        """
        Args:
          data_dir: `string`, the path to the folder containing the files in the
            downloaded .tar
          citation: `string`, citation for the data set
          url: `string`, url for information about the data set
          **kwargs: keyword arguments forwarded to super.
        """
        super(EgyptianSpeechCorpusConfig, self).__init__(version=datasets.Version("2.1.0", ""), **kwargs)


def map_to_array(batch):
    start, stop = batch["segment"].split("_")
    speech_array, _ = sf.read(batch["file"], start=start, stop=stop)
    batch["speech"] = speech_array
    return batch


class EgyptionSpeechCorpus(datasets.GeneratorBasedBuilder):
    """EgyptianSpeechCorpus dataset."""

    BUILDER_CONFIGS = [
        EgyptianSpeechCorpusConfig(name="clean", description="'Clean' speech."),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "file": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "segment": datasets.Value("string"),
                }
            ),
            supervised_keys=("file", "text"),
            homepage=_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        self.archive_path = "/content/mgb3"
        return [
            datasets.SplitGenerator(
                name="train", gen_kwargs={"archive_path": os.path.join(self.archive_path, "adapt")}
            ),
            datasets.SplitGenerator(name="dev", gen_kwargs={"archive_path": os.path.join(self.archive_path, "dev")}),
            datasets.SplitGenerator(name="test", gen_kwargs={"archive_path": os.path.join(self.archive_path, "test")}),
        ]

    def _generate_examples(self, archive_path):
        """Generate examples from a Librispeech archive_path."""
        text_dir = os.path.join(archive_path, "Alaa")
        wav_dir = os.path.join(self.archive_path, "wav")

        segments_file = os.path.join(text_dir, "text_noverlap")

        with open(segments_file, "r", encoding="utf-8") as f:
            for _id, line in enumerate(f):
                segment = line.split(" ")[0]
                text = " ".join(line.split(" ")[1:])
                wav_file = "_".join(segment.split("_")[:4]) + ".wav"
                start, stop = segment.split("_")[4:6]
                wav_path = os.path.join(wav_dir, wav_file)
                if (wav_file in corrupt_files) or (wav_file not in os.listdir(wav_dir)):
                    continue
                example = {"file": wav_path, "text": text, "segment": ("_").join([start, stop])}
                yield str(_id), example

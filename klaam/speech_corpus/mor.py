"""Arabic Speech Corpus"""

from __future__ import absolute_import, division, print_function

import os

import datasets

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


class MorrocanSpeechCorpusConfig(datasets.BuilderConfig):
    """BuilderConfig for MorrocanSpeechCorpus."""

    def __init__(self, **kwargs):
        """
        Args:
          data_dir: `string`, the path to the folder containing the files in the
            downloaded .tar
          citation: `string`, citation for the data set
          url: `string`, url for information about the data set
          **kwargs: keyword arguments forwarded to super.
        """
        super(MorrocanSpeechCorpusConfig, self).__init__(version=datasets.Version("2.1.0", ""), **kwargs)


class MorrocanSpeechCorpus(datasets.GeneratorBasedBuilder):
    """MorrocanSpeechCorpus dataset."""

    BUILDER_CONFIGS = [
        MorrocanSpeechCorpusConfig(name="clean", description="'Clean' speech."),
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
        self.archive_path = "/content/MGB5 Moroccan"
        return [
            datasets.SplitGenerator(
                name="train", gen_kwargs={"archive_path": os.path.join(self.archive_path, "train")}
            ),
            datasets.SplitGenerator(name="dev", gen_kwargs={"archive_path": os.path.join(self.archive_path, "dev")}),
            datasets.SplitGenerator(name="test", gen_kwargs={"archive_path": os.path.join(self.archive_path, "test")}),
        ]

    def _generate_examples(self, archive_path):
        """Generate examples from a Librispeech archive_path."""
        text_dir = os.path.join(archive_path, "all_files")
        wav_dir = os.path.join(archive_path, "wav")

        for text_file in os.listdir(text_dir):
            if text_file.endswith(".txt"):
                segments_file = os.path.join(text_dir, text_file)
                with open(segments_file, "r", encoding="utf-8") as f:
                    for _id, line in enumerate(f):
                        segment = line.split(" ")[0]
                        text = " ".join(line.split(" ")[1:])
                        wav_file = "_".join(segment.split("_")[:2]) + ".wav"
                        start, stop = segment.split("_")[4:6]
                        wav_path = os.path.join(wav_dir, wav_file)
                        example = {"file": wav_path, "text": text, "segment": ("_").join([start, stop])}
                        yield str(_id), example

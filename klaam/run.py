import torch
import yaml
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from klaam.external.FastSpeech2.inference import infer_tts, prepare_tts_model
from klaam.models.wav2vec import Wav2Vec2ClassificationModel
from klaam.processors.wav2vec import CustomWav2Vec2Processor
from klaam.utils.utils import load_file_to_data, predict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SpeechRecognition:
    def __init__(self, lang="egy", path=None):
        self.lang = lang
        self.bw = False
        if path is None:
            if lang == "egy":
                model_dir = "Zaid/wav2vec2-large-xlsr-53-arabic-egyptian"
            elif lang == "msa":
                model_dir = "elgeish/wav2vec2-large-xlsr-53-arabic"
                self.bw = True
        else:
            model_dir = path
        self.model = Wav2Vec2ForCTC.from_pretrained(model_dir).to(DEVICE)
        self.processor = Wav2Vec2Processor.from_pretrained(model_dir)

    def transcribe(self, wav_file):

        return predict(load_file_to_data(wav_file), self.model, self.processor, mode="rec", bw=self.bw)


class SpeechClassification:
    def __init__(self, path=None):
        if path is None:
            dir = "Zaid/wav2vec2-large-xlsr-dialect-classification"
        else:
            dir = path
        self.model = Wav2Vec2ClassificationModel.from_pretrained(dir).to(DEVICE)
        self.processor = CustomWav2Vec2Processor.from_pretrained(dir)

    def classify(self, wav_file, return_prob=False):
        return predict(load_file_to_data(wav_file), self.model, self.processor, mode="cls", return_prob=return_prob)


class TextToSpeech:
    def __init__(
        self,
        prepare_tts_model_path,
        model_config_path,
        train_config_path,
        vocoder_config_path=None,
        speaker_pre_trained_path=None,
        root_path=None,
    ):
        self.prepare_tts_model = yaml.load(open(prepare_tts_model_path, "r"), Loader=yaml.FullLoader)
        # TODO: fix this trick
        if self.prepare_tts_model["path"]["stats_path"][0] != "/":
            self.prepare_tts_model["path"]["stats_path"] = f"{root_path}/{self.prepare_tts_model['path']['stats_path']}"
        if self.prepare_tts_model["path"]["lexicon_path"][0] != "/":
            self.prepare_tts_model["path"][
                "lexicon_path"
            ] = f"{root_path}/{self.prepare_tts_model['path']['lexicon_path']}"
        self.model_config = yaml.load(open(model_config_path, "r"), Loader=yaml.FullLoader)
        self.train_config = yaml.load(open(train_config_path, "r"), Loader=yaml.FullLoader)
        self.configs = (self.prepare_tts_model, self.model_config, self.train_config)
        self.vocoder_config_path = vocoder_config_path
        self.speaker_pre_trained_path = speaker_pre_trained_path
        self.model, self.vocoder, self.configs = prepare_tts_model(
            self.configs, self.vocoder_config_path, self.speaker_pre_trained_path
        )

    def synthesize(self, text, bw=False, apply_tshkeel=False):
        infer_tts(text, self.model, self.vocoder, self.configs, bw=bw, apply_tshkeel=apply_tshkeel)

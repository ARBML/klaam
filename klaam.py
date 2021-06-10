from transformers import (Wav2Vec2ForCTC,Wav2Vec2Processor)
from utils import load_file_to_data, predict
from models import Wav2Vec2ClassificationModel
from processors import CustomWav2Vec2Processor
from FastSpeech2.inference import prepare_tts_model, infer_tts
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SpeechRecognition:

    def __init__(self, lang = 'egy', path = None):
        self.lang = lang
        self.bw = False
        if path is None:
            if lang == 'egy':
                model_dir = 'Zaid/wav2vec2-large-xlsr-53-arabic-egyptian'
            elif lang == 'msa':
                model_dir = 'elgeish/wav2vec2-large-xlsr-53-arabic'
                self.bw = True
        else:
            model_dir = path
        self.model = Wav2Vec2ForCTC.from_pretrained(model_dir).to(device)
        self.processor = Wav2Vec2Processor.from_pretrained(model_dir)
    
    def transcribe(self, wav_file):
        
        return predict(load_file_to_data(wav_file), 
                self.model, self.processor, mode = 'rec', bw = self.bw)

class SpeechClassification:

    def __init__(self, path = None):
        if path is None:
            dir = 'Zaid/wav2vec2-large-xlsr-dialect-classification'
        else:
            dir = path
        self.model = Wav2Vec2ClassificationModel.from_pretrained(dir).to(device)
        self.processor = CustomWav2Vec2Processor.from_pretrained(dir)
    
    def classify(self, wav_file, return_prob = False):
        return predict(load_file_to_data(wav_file), 
                    self.model, self.processor, mode = 'cls', return_prob = return_prob)

class TextToSpeech:

    def __init__(self):
        self.model, self.vocoder, self.configs = prepare_tts_model()
    
    def synthesize(self, text, bw = False, apply_tshkeel = False):
        infer_tts(text, self.model, self.vocoder, self.configs, bw = bw, apply_tshkeel = apply_tshkeel)

import re
import argparse
from string import punctuation
from .phonetise.phonetise_arabic import phonetise
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from pypinyin import pinyin, Style
from .buckwalter import ar2bw, bw2ar
from .utils.model import get_model_inference, get_vocoder
from .utils.tools import to_device, synth_samples
from .dataset import TextDataset
from .text import text_to_sequence
import mishkal.tashkeel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def preprocess_arabic(text, preprocess_config, bw = False, ts = False):

    if bw:
        text = "".join([bw2ar[l] if l in bw2ar else l for l in text])
    
    if ts:
        vocalizer = mishkal.tashkeel.TashkeelClass()
        text = vocalizer.tashkeel(text).strip()

    phones = phonetise(text)[0]   
    phones = "{" + phones + "}"

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        #TO_DO
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)

def synthesize(model, step, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
            )


def prepare_tts_model():
    # Read Config
    preprocess_config = yaml.load(
        open("FastSpeech2/config/Arabic/preprocess.yaml", "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open('FastSpeech2/config/Arabic/model.yaml', "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open('FastSpeech2/config/Arabic/train.yaml', "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model_inference(configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)
    return model, vocoder, configs


def infer_tts(text, model, vocoder, configs, bw = True, apply_tshkeel = False, 
             pitch_control = 1.0, energy_control = 1.0, duration_control = 1.0):
    control_values = pitch_control, energy_control, duration_control
    (preprocess_config, _, _) = configs 
    ids = raw_texts = [text[:100]]
    speakers = np.array([0])
    texts = np.array([preprocess_arabic(text, preprocess_config, bw = bw, ts = apply_tshkeel)])
    text_lens = np.array([len(texts[0])])
    batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]
    synthesize(model, '', configs, vocoder, batchs, control_values)

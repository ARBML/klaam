import mishkal.tashkeel
import numpy as np
import torch

from klaam.external.FastSpeech2.buckwalter import bw2ar
from klaam.external.FastSpeech2.phonetise.phonetise_arabic import phonetise
from klaam.external.FastSpeech2.text import text_to_sequence
from klaam.external.FastSpeech2.utils.model import get_model_inference, get_vocoder
from klaam.external.FastSpeech2.utils.tools import synth_samples, to_device

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_arabic(text, preprocess_config, bw=False, ts=False):

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
        # TO_DO
        text_to_sequence(phones, preprocess_config["preprocessing"]["text"]["text_cleaners"])
    )

    return np.array(sequence)


def synthesize(model, step, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    for batch in batchs:
        batch = to_device(batch, DEVICE)
        with torch.no_grad():
            # Forward
            output = model(*(batch[2:]), p_control=pitch_control, e_control=energy_control, d_control=duration_control)
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
            )


def prepare_tts_model(configs, vocoder_config_path, speaker_pre_trained_path):
    model_config = configs[1]

    # Get model
    model = get_model_inference(configs, DEVICE, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, DEVICE, vocoder_config_path, speaker_pre_trained_path)
    return model, vocoder, configs


def infer_tts(
    text,
    model,
    vocoder,
    configs,
    bw=True,
    apply_tshkeel=False,
    pitch_control=1.0,
    energy_control=1.0,
    duration_control=1.0,
):
    control_values = pitch_control, energy_control, duration_control
    (preprocess_config, _, _) = configs
    ids = raw_texts = [text[:100]]
    speakers = np.array([0])
    texts = np.array([preprocess_arabic(text, preprocess_config, bw=bw, ts=apply_tshkeel)])
    text_lens = np.array([len(texts[0])])
    batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]
    synthesize(model, "", configs, vocoder, batchs, control_values)

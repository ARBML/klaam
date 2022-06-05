# klaam
Arabic speech recognition, classification and text-to-speech using many advanced models like wave2vec and fastspeech2. This repository allows training and prediction using pretrained models.

<p align="center">
<img src="https://raw.githubusercontent.com/ARBML/klaam/main/misc/klaam_logo.png" width="250px"/>
</p>


## Usage

### Speech Classification
```python
from klaam import SpeechClassification
model = SpeechClassification()
model.classify(wav_file)
```

### Speech Recongnition
```python
from klaam import SpeechRecognition
model = SpeechRecognition()
model.transcribe(wav_file)
```


### Text To Speech
```python
from klaam import TextToSpeech
prepare_tts_model_path = "../cfgs/FastSpeech2/config/Arabic/preprocess.yaml"
model_config_path = "../cfgs/FastSpeech2/config/Arabic/model.yaml"
train_config_path = "../cfgs/FastSpeech2/config/Arabic/train.yaml"
vocoder_config_path = "../cfgs/FastSpeech2/model_config/hifigan/config.json"
speaker_pre_trained_path = "../data/model_weights/hifigan/generator_universal.pth.tar"

model = TextToSpeech(prepare_tts_model_path, model_config_path, train_config_path, vocoder_config_path, speaker_pre_trained_path)

model.synthesize(sample_text)
```

There are two avilable models for recognition trageting MSA and egyptian dialect . You can set any of them using the `lang` attribute

```python
from klaam import SpeechRecognition
model = SpeechRecognition(lang = 'msa')
model.transcribe('file.wav')
```

## Datasets

| Dataset | Description | link |
|---------| ------------------------------ | ---- |
| MGB-3  | Egyptian Arabic Speech recognition in the wild. Every sentence was annotated by four annotators. More than 15 hours have been collected from YouTube.  |  requires registeration [here](https://arabicspeech.org/mgb3-asr/)|
| ADI-5  | More than 50 hours collected from Aljazeera TV.  4 regional dialectal: Egyptian (EGY), Levantine (LAV), Gulf (GLF), North African (NOR), and Modern Standard Arabic (MSA). This dataset is a part of the MGB-3 challenge.  | requires registeration [here](https://arabicspeech.org/mgb3-adi/)|
|Common voice | Multlilingual dataset avilable on huggingface | [here](https://github.com/huggingface/datasets/tree/master/datasets/common_voice). |
|Arabic Speech Corpus | Arabic dataset with alignment and transcriptions | [here](http://en.arabicspeechcorpus.com/). |

## Models

We currently support four models, three of them are avilable on transformers.

|Language | Description | Source |
|-------- | ----------- | ------ |
|Egyptian | Speech recognition | [wav2vec2-large-xlsr-53-arabic-egyptian](https://huggingface.co/Zaid/wav2vec2-large-xlsr-53-arabic-egyptian)|
|Standard Arabic | Speech recognition | [wav2vec2-large-xlsr-53-arabic](https://huggingface.co/elgeish/wav2vec2-large-xlsr-53-arabic)
|EGY, NOR, LAV, GLF, MSA | Speech classification | [wav2vec2-large-xlsr-dialect-classification](https://huggingface.co/Zaid/wav2vec2-large-xlsr-dialect-classification)|
|Standard Arabic| Text-to-Speech | [fastspeech2]()|

## Example Notebooks
<table>
  <tr>
    <th><b>Name</b></th>
    <th><b>Description</b></th>
    <th><b>Notebook</b></th>
  </tr>

  <tr>
    <td>Demo</td>
    <td>Classification, Recongition and Text-to-speech  in a few lines of code.</td>
    <td><a href="https://colab.research.google.com/github/ARBML/klaam/blob/main/notebooks/demo.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg"  >
    </a></td>
  </tr>

  <tr>
    <td>Demo with mic</td>
    <td>Audio Recongition and classification with recording.</td>
    <td><a href="https://colab.research.google.com/github/ARBML/klaam/blob/main/notebooks/demo_with_mic.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg">
    </a></td>
  </tr>
<table>

## Training

The scripts are a modification of [jqueguiner/wav2vec2-sprint](https://github.com/jqueguiner/wav2vec2-sprint).

### classification
This script is used for the classification task on the 5 classes.

```sh
python run_classifier.py \
    --model_name_or_path="facebook/wav2vec2-large-xlsr-53" \
    --output_dir=/path/to/output \
    --cache_dir=/path/to/cache/ \
    --freeze_feature_extractor \
    --num_train_epochs="50" \
    --per_device_train_batch_size="32" \
    --preprocessing_num_workers="1" \
    --learning_rate="3e-5" \
    --warmup_steps="20" \
    --evaluation_strategy="steps"\
    --save_steps="100" \
    --eval_steps="100" \
    --save_total_limit="1" \
    --logging_steps="100" \
    --do_eval \
    --do_train \
```

### Recognition

This script is for training on the dataset for pretraining on the egyption dialects dataset.

```sh
python run_mgb3.py \
    --model_name_or_path="facebook/wav2vec2-large-xlsr-53" \
    --output_dir=/path/to/output \
    --cache_dir=/path/to/cache/ \
    --freeze_feature_extractor \
    --num_train_epochs="50" \
    --per_device_train_batch_size="32" \
    --preprocessing_num_workers="1" \
    --learning_rate="3e-5" \
    --warmup_steps="20" \
    --evaluation_strategy="steps"\
    --save_steps="100" \
    --eval_steps="100" \
    --save_total_limit="1" \
    --logging_steps="100" \
    --do_eval \
    --do_train \
```

This script can be used for Arabic common voice training

```sh
python run_common_voice.py \
    --model_name_or_path="facebook/wav2vec2-large-xlsr-53" \
    --dataset_config_name="ar" \
    --output_dir=/path/to/output/ \
    --cache_dir=/path/to/cache \
    --overwrite_output_dir \
    --num_train_epochs="1" \
    --per_device_train_batch_size="32" \
    --per_device_eval_batch_size="32" \
    --evaluation_strategy="steps" \
    --learning_rate="3e-4" \
    --warmup_steps="500" \
    --fp16 \
    --freeze_feature_extractor \
    --save_steps="10" \
    --eval_steps="10" \
    --save_total_limit="1" \
    --logging_steps="10" \
    --group_by_length \
    --feat_proj_dropout="0.0" \
    --layerdrop="0.1" \
    --gradient_checkpointing \
    --do_train --do_eval \
    --max_train_samples 100 --max_val_samples 100
```

### Text To Speech

We use the pytorch implementation of fastspeech2 by [ming024](https://github.com/ming024/FastSpeech2). The procedure is as follows

Download the dataset

```
wget http://en.arabicspeechcorpus.com/arabic-speech-corpus.zip
unzip arabic-speech-corpus.zip
```

Create multiple directories for data

```
mkdir -p raw_data/Arabic/Arabic preprocessed_data/Arabic/TextGrid/Arabic
cp arabic-speech-corpus/textgrid/* preprocessed_data/Arabic/TextGrid/Arabic
```

Prepare metadata

```python
import os
base_dir = '/content/arabic-speech-corpus'
lines = []
for lab_file in os.listdir(f'{base_dir}/lab'):
  lines.append(lab_file[:-4]+'|'+open(f'{base_dir}/lab/{lab_file}', 'r').read())


open(f'{base_dir}/metadata.csv', 'w').write(('\n').join(lines))
```

Clone my fork

```bash
git clone --depth 1 https://github.com/zaidalyafeai/FastSpeech2
cd FastSpeech2
pip install -r requirements.txt
```

Prepare alignments and prepreocessed data

```
python3 prepare_align.py config/Arabic/preprocess.yaml
python3 preprocess.py config/Arabic/preprocess.yaml
```

Unzip vocoders

```
unzip hifigan/generator_LJSpeech.pth.tar.zip -d hifigan
unzip hifigan/generator_universal.pth.tar.zip -d hifigan
```

Start training

```
python3 train.py -p config/Arabic/preprocess.yaml -m config/Arabic/model.yaml -t config/Arabic/train.yaml
```

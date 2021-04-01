# klaam
Arabic speech recognition and classification

 <p align="center"> 
 <img src = "https://raw.githubusercontent.com/ARBML/klaam/main/klaam_logo.PNG" width = "250px"/>
 </p>
 
 ## Usage 
 
 ```python
 from klaam import SpeechClassification
 model = SpeechClassification()
 model.classify('file.wav')
 
 from klaam import SpeechRecognition
 model = SpeechRecognition()
 model.transcribe('file.wav')
 ```
 
 ## Datasets 
 
| Dataset | Description | link |
|---------| ------------------------------ | ---- |
| MGB-3  | Egyptian Arabic Speech recognition in the wild. Every sentence was annotated by four annotators. More than 15 hours have been collected from YouTube.  |  requires registeration [here](https://arabicspeech.org/mgb3-asr/)|
| ADI-5  | More than 50 hours collected from Aljazeera TV.  4 regional dialectal: Egyptian (EGY), Levantine (LAV), Gulf (GLF), North African (NOR), and Modern Standard Arabic (MSA). This dataset is a part of the MGB-3 challenge.  | requires registeration [here](https://arabicspeech.org/mgb3-adi/)|
|Common voice | Multlilingual dataset avilable on huggingface | [here](https://github.com/huggingface/datasets/tree/master/datasets/common_voice). |

## Training

### classification 
```
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

```
python run_recognition.py \
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
## Example Notebooks 
<table class="tg">

  <tr>
    <th class="tg-yw4l"><b>Name</b></th>
    <th class="tg-yw4l"><b>Description</b></th>
    <th class="tg-yw4l"><b>Notebook</b></th>
  </tr>

  <tr>
    <td class="tg-yw4l">Speech Classification</td>
    <td class="tg-yw4l">Classify input into either MSA, GULF, North Africa, Egypt or Levantine </td>
    <td class="tg-yw4l"><a href="https://colab.research.google.com/github/ARBML/klaam/blob/main/demo.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg"  >
    </a></td>
  </tr>
<table>


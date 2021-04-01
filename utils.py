import soundfile as sf
import torch

def load_file_to_data(file):
    batch = {}
    start = 0 
    stop = 20 
    srate = 16_000
    speech, sampling_rate = sf.read(file, start = start * srate , stop = stop * srate)
    batch["speech"] = speech
    batch["sampling_rate"] = sampling_rate
    return batch


def predict(data, processor, model):
    features = processor(data["speech"], 
                        sampling_rate=data["sampling_rate"],
                        max_length=320000,
                        pad_to_multiple_of=320000,
                        padding=True, return_tensors="pt")
    
    input_values = {'input_values':features.input_values.to("cuda")}
    attention_mask = features.attention_mask.to("cuda")
    with torch.no_grad():
        outputs = model(**input_values, attention_mask=attention_mask)
    pred_id = torch.argmax(outputs['logits'], dim=-1)[0]
    return pred_id
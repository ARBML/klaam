import soundfile as sf
import torch

def load_file_to_data(file, max_len = 20, srate = 16_000):
    batch = {} 
    speech, sampling_rate = sf.read(file, start = 0 , stop = max_len * srate)
    batch["speech"] = speech
    batch["sampling_rate"] = sampling_rate
    return batch


def predict(data, processor, model, mode = 'rec'):
    if 'rec':
        features = processor(data["speech"], sampling_rate=data["sampling_rate"], padding=True, return_tensors="pt")
    else:
        features = processor(data["speech"], 
                        sampling_rate=data["sampling_rate"],
                        max_length=320000,
                        pad_to_multiple_of=320000,
                        padding=True, return_tensors="pt")
    
    input_values = features.input_values.to("cuda")
    attention_mask = features.attention_mask.to("cuda")
    with torch.no_grad():
        outputs = model(input_values, attention_mask=attention_mask)
    
    
    if mode == 'rec':
        pred_ids = torch.argmax(outputs.logits, dim=-1)[0]
        return processor.batch_decode(pred_ids)
    else:
        pred_ids = torch.argmax(outputs['logits'], dim=-1)[0]
        return pred_ids
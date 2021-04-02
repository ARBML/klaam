import torch
import librosa    


def load_file_to_data(file, max_len = 20, srate = 16_000):
    batch = {} 
    speech, sampling_rate = y, s = librosa.load('test.wav', sr=srate)
    batch["speech"] = speech
    batch["sampling_rate"] = sampling_rate
    return batch


def predict(data, model, processor, mode = 'rec'):
    if mode == 'rec':
        features = processor(data["speech"],
                            sampling_rate=data["sampling_rate"],
                            padding=True,
                            max_length=128000, 
                            pad_to_multiple_of=128000,
                            return_tensors="pt")
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
        pred_ids = torch.argmax(outputs.logits, dim=-1)
        return processor.batch_decode(pred_ids)
    else:
        pred_ids = torch.argmax(outputs['logits'], dim=-1)[0]
        dialects = ['EGY','NOR','GLF','LAV','MSA']
        return dialects[pred_ids]
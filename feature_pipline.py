import os
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from datasets import Audio

class feature_pip():
    def __init__(self, common_voice_data):
        self.commmon_voice = common_voice_data
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
        self.tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="zh-CN", task="transcribe")
        
    def prepare_dataset(self, batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array
        batch["input_features"] = self.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids
        batch["labels"] = self.tokenizer(batch["sentence"]).input_ids
        return batch
    
    def pipline(self):
        # Remove irrelevant columns in the data
        self.common_voice = self.common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])

        # Set the audio inputs to the correct sampling rate
        self.common_voice = self.common_voice.cast_column("audio", Audio(sampling_rate=16000))

        self.common_voice = self.common_voice.map(self.prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=2)



        
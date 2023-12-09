import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from transformers import WhisperProcessor
from transformers import WhisperTokenizer
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
    
class Trainer():
    def __init__(self, processed_data, learning_rate, warmup_steps, max_steps, save_steps, eval_steps):
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="zh-CN", task="transcribe")
        self.metric = evaluate.load("wer")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

        self.common_vioce_processed = processed_data
        self.lr = learning_rate
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps

    
    def compute_metrics(self, pred):
        tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="zh-CN", task="transcribe")

        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * self.metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
    
    def train(self):
        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []

        # Training configuration
        training_args = Seq2SeqTrainingArguments(
            output_dir="./whisper-small-zh", 
            per_device_train_batch_size=16,
            gradient_accumulation_steps=1,  
            learning_rate=self.lr,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
            gradient_checkpointing=True,
            fp16=True,
            evaluation_strategy="steps",
            per_device_eval_batch_size=8,
            predict_with_generate=True,
            generation_max_length=225,
            save_steps=self.save_steps,
            eval_steps=self.eval_steps,
            logging_steps=25,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            push_to_hub=True,
        )

        # Trainer
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)
        s2s_trainer = Seq2SeqTrainer(
            args=training_args,
            model=self.model,
            train_dataset=self.common_voice_processed["train"],
            eval_dataset=self.common_voice_processed["test"],
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=self.processor.feature_extractor,
        )

        self.processor.save_pretrained(training_args.output_dir)

        s2s_trainer.train()

        # Push the trainer model to hugging face
        kwargs = {
            "dataset_tags": "mozilla-foundation/common_voice_11_0",
            "dataset": "Common Voice 11.0",  # a 'pretty' name for the training dataset
            "dataset_args": "config: zh, split: test",
            "language": "zh",
            "model_name": "Whisper Small zh",  
            "finetuned_from": "openai/whisper-small",
            "tasks": "automatic-speech-recognition",
            "tags": "hf-asr-leaderboard",
        }
        s2s_trainer.push_to_hub(**kwargs)
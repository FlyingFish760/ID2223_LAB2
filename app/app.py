import gradio as gr
import pytube as pt
from transformers import pipeline
import os
from huggingface_hub import HfFolder
from gtts import gTTS

# Initialize pipelines for transcriptionn and translation
transcription_pipe = pipeline(model="kaidiXu/whisper-small-zh", token=HfFolder.get_token())
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en")


def process_audio(file_path):
    text = transcription_pipe(file_path)["text"]
    translation = translator(text)[0]["translation_text"]
    return text, translation


def download_youtube_audio(yt_url):
    yt = pt.YouTube(yt_url)
    stream = yt.streams.filter(only_audio=True).first()
    file_path = stream.download(filename="temp_audio.mp3")
    return file_path


def youtube_transcription(yt_url):
    audio_path = download_youtube_audio(yt_url)
    results = process_audio(audio_path)
    os.remove(audio_path)  # Clean up the downloaded file
    return results


def transcribe_and_process(rec=None, file=None):
    if rec is not None:
        audio = rec
    elif file is not None:
        audio = file
    else:
        return "Provide a recording or a file."

    return process_audio(audio)


app = gr.Blocks()

# Gradio interface
with app:
    gr.Markdown('<div style="text-align:center"><h2>Whisper Small Chinese</h2></div>')
    gr.Markdown("Real-time demo for Chinese speech recognition using a fine-tuned Whisper small model.")

    with gr.Tab("Audio"):
        with gr.Row():
            audio_input = gr.Audio(sources="microphone", label="Speak into the microphone", type="filepath")
            audio_process_button = gr.Button("Audio to Transcription and Translation")
            audio_transcription, audio_translation = gr.Textbox(label="Transcription"), gr.Textbox(label="Translation")
        audio_process_button.click(fn=transcribe_and_process, inputs=audio_input,
                                   outputs=[audio_transcription, audio_translation])

    with gr.Tab("YouTube"):
        with gr.Row():
            yt_input = gr.Textbox(label="Paste YouTube URL here")
            yt_process_button = gr.Button("YouTube Video to Transcription and Translation")
        yt_transcription, yt_translation = gr.Textbox(label="Transcription"), gr.Textbox(label="Translation")
        yt_process_button.click(fn=youtube_transcription, inputs=yt_input,
                                outputs=[yt_transcription, yt_translation])

(app.launch())
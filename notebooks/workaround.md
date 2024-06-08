# Overcoming OpenAI API Limitations with Local GPU Acceleration for Audio Transcription

## Introduction

This document chronicles my journey through a challenge encountered in a Langchain course offered by DeepLearning.AI. The course recommended using OpenAI's API for audio transcription, a resource I neither had access to nor could use locally on my machine. Faced with this limitation, I embarked on a quest to find a workaround that allowed me to perform the transcription locally, utilizing my NVIDIA GeForce GTX 1650 GPU. This narrative covers the problem, the solution I devised, and the invaluable lessons learned along the way.

## The Problem

In the context of the course, the OpenAIWhisperParser API was utilized for transcribing audio from YouTube videos. However, this API is part of OpenAI's paid services, which was not a feasible option for me. Additionally, the necessity for a local solution was paramount, as my objective was to utilize my GPU to its fullest potential, thereby accelerating the transcription process.

## The Solution

### Step 1: Preparing the Local Environment

#### Setting up CUDA and cuDNN

To harness the power of my NVIDIA GeForce GTX 1650 GPU, setting up the CUDA environment was imperative. This involved:

1. **Installing the NVIDIA Driver**: Ensuring compatibility with the GTX 1650.
2. **Downloading and Installing the CUDA Toolkit**: Selecting a version compatible with the PyTorch installation.
3. **Installing cuDNN**: Integrating it with the CUDA setup to optimize deep learning tasks.
4. **Verifying the Installation**: Confirming that CUDA was recognized and operational on my system.

#### Installing PyTorch with CUDA Support

With the environment ready, the next step was to install PyTorch configured for CUDA:

```shell
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

Validating CUDA Integration with PyTorch
Ensuring PyTorch's ability to leverage GPU acceleration:

```python
import torch
print(torch.cuda.is_available())  # Expected output: True
```

### Step 2: Local Audio Transcription

With the API out of reach, the focus shifted to local solutions. This involved downloading the YouTube video's audio and utilizing the Whisper model for transcription:

1. **Downloading the Video's Audio**: Utilizing yt_dlp and pydub for audio extraction.

```python
! pip install yt_dlp
! pip install pydub
! pip install langchain
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers.audio import OpenAIWhisperParserLocal
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

url="https://www.youtube.com/watch?v=jGwO_UgTS7I"
save_dir="../../docs/youtube/"

YoutubeAudioLoader([url],save_dir)

loader = GenericLoader(
    YoutubeAudioLoader([url],save_dir),
    OpenAIWhisperParser()
)
docs = loader.load()
```
`Note`: This code will return an error because it's necessary an Openai API key valid. This step is only to download and exctrat the audio from the Youtube video.


2. **Performing Local Transcription with Whisper**: Transcribing the audio file and saving the results in a JSON format for further processing.

```python
! pip install -U openai-whisper
import json
import whisper

model = whisper.load_model("base", device="cuda")
audio_path = "../../docs/youtube/Stanford CS229： Machine Learning Course, Lecture 1 - Andrew Ng (Autumn 2018).m4a"
result = model.transcribe(audio_path, language = "en", fp16=False)

# Path to save transcript
path_youtube_transcript = "../../docs/youtube/transcript.json"

# Serializa transcript to JSON
with open(path_youtube_transcript, 'w', encoding='utf-8') as arquivo:
    json.dump(result, arquivo, ensure_ascii=False, indent=4)
```

3. **Using transcript JSON to make Langchain Document**: Now it's possible to transform one Youtube video into a Langchain document.

```python
!pip install jq

from langchain_community.document_loaders import JSONLoader

loader = JSONLoader(
    file_path=path_youtube_transcript,
    jq_schema='.segments[].text',
    text_content=False)

data = loader.load()
```


### Step 3: A Simplified Alternative

After navigating through the complex setup and local transcription process, a simpler alternative emerged. The YouTube transcripts doc loader offered an efficient and easier path to the same goal:

```python
! pip install --upgrade --quiet youtube-transcript-api
! pip install --upgrade --quiet pytube
from langchain_community.document_loaders import YoutubeLoader

loader = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=jGwO_UgTS7I",
    add_video_info=True,
    language=["en", "id"],
    translation="en",
)
docs = loader.load()
```

## Lessons Learned
- **Thorough Documentation Review**: Before delving into complex solutions, exploring official documentation and simpler alternatives can save time and effort.
- **Leveraging Local GPU for Efficiency**: The experience underscored the value of setting up and utilizing local GPU resources for tasks requiring significant computational power, such as audio transcription.

## Conclusion
This journey from facing a restrictive limitation to discovering an efficient, local solution has been both challenging and enlightening. Not only did it enhance my understanding of CUDA setup and local GPU utilization, but it also highlighted the importance of seeking simpler solutions and the vast potential of open-source tools in overcoming obstacles.
### Emotion Recognition from Audio Transcripts ###


This project performs emotion classification on transcribed audio files using fine-tuned Transformer models (DistilBERT and DistilRoBERTa). It transcribes speech into text using the AssemblyAI API and classifies each sentence into one of seven emotions.

## Project Structure ##
pipeline(model, audio_file_path)

The main function that:

 - Transcribes the audio file.

 - Loads the appropriate model (dbert or droberta).

 - Tokenizes the text.

 - Returns a DataFrame with start/end timestamps and predicted emotions.

## Models ##
Supported Models:

 - dbert → DistilBERT fine-tuned for 7-class emotion classification

 - droberta → DistilRoBERTa fine-tuned for 7-class emotion classification

## Emotion Labels ##
 - neutral

 - surprise

 - disgust

 - sadness

 - happiness

 - anger

 - fear

## Model weights are stored at ##

../Task_5/DistilBERT/dbert_iter4_weights.h5

../Task_5/DistillRoberta/droberta_iter2_weights.h5

## API Key ##
To use the transcription feature, set your AssemblyAI API key in the script:

aai.settings.api_key = "your_api_key_here"
Alternatively, you can export it as an environment variable.

## Usage ##

    from your_script import pipeline
    
    audio_path = "path/to/audio.wav"
    df = pipeline('droberta', audio_path)
    print(df)
    Output:
    Text	Start Time (s)	End Time (s)	Emotion
    "I'm so excited to be here!"	0.00	1.92	happiness
    "I can't believe this happened"	2.05	4.18	surprise
    ...	...	...	...

## Notes ##
This script assumes you are working in a structured folder (e.g., Task_2, Task_5, etc.).

Ensure that the weights match the models used in training.

Tokenizers used are from Hugging Face's Transformers, matching the pre-trained models.
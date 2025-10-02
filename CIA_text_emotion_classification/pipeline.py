import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification, AutoTokenizer, AutoConfig, TFAutoModelForSequenceClassification
from keras.layers import TFSMLayer
import sys
import assemblyai as aai
import csv
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from datetime import datetime
import logging

# Suppress HTTP request logs
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

sys.path.append(r'home/y2c/2024-25c-fai2-adsai-group-group_11_2c/Task_2')

### HELPER FUNCTIONS ###

# Loading model and tokenizer
def loading_model(model_type):
    if model_type == 'dbert':
        model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=7)
        model.load_weights('../Task_5/DistilBERT/dbert_iter4_weights.h5')

        return model
        
    elif model_type == 'droberta':
        # Load the model configuration
        config = AutoConfig.from_pretrained('j-hartmann/emotion-english-distilroberta-base', num_labels=7)
        
        # Modify the dropout rate in the configuration
        config.attention_probs_dropout_prob = 0.5  # Set the dropout rate for attention layers
        config.hidden_dropout_prob = 0.5  # Set the dropout rate for hidden layers
        
        # Using larger dropout and weight decay to prevent overfitting on augmented data
        model = TFAutoModelForSequenceClassification.from_pretrained(
            'j-hartmann/emotion-english-distilroberta-base',
            config = config
        )

        model.load_weights('../Task_5/DistillRoberta/droberta_iter2_weights.h5')

        return model
        
    else:
        raise ValueError("Invalid model_type. Expected 'dbert' or 'droberta'. Got: {}".format(model_type))
# Changed task 2 for returning a dataframe without saving it into a csv file
def transcribe_audio_to_df(audio_file_path):
    # Set your API key
    aai.settings.api_key = "fb2df8accbcb4f38ba02666862cd6216"

    # Transcribe audio
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_file_path)

    # Extract sentences into a list of dicts
    data = []
    for i, sentence in enumerate(transcript.get_sentences(), 1):
        data.append({
            "Text": sentence.text,
            "Start Time (s)": round(sentence.start / 1000, 2),
            "End Time (s)": round(sentence.end / 1000, 2),
        })

    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df

def tokenize_texts(texts, tokenizer):
    return tokenizer(list(texts), truncation=True, padding=True, return_tensors="tf", max_length=128)

def pipeline(model, audio_file_path):

    valid_models = ['dbert', 'droberta']
    unique_labels = ['neutral', 'surprise', 'disgust', 'sadness', 'happiness', 'anger', 'fear']

    # Initialize and fit the LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(unique_labels)
    
    # Check if the passed model is valid
    if model not in valid_models:
        raise ValueError(f"Invalid model! Expected one of {valid_models}, but got '{model}'")

    ### 2. TRANSCRIBING AUDIO DATA (TASK 2) ###
    print("Exctracting the transcript of the audio...")
    df = transcribe_audio_to_df(audio_file_path)

    ### 3. Loading the models based on input
    if model == 'dbert':
        print(f"Initializing {model}...")
        model = loading_model(model)
        
        # Since I did not change the tokenizer in the training by adding custom tokens or changing the vocabulary size, it is fine to use the base one from Transformers.
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        df_tokenized = tokenize_texts(df['Text'].tolist(), tokenizer)

        # Prepare the tokenized input as a dictionary for the model
        inputs = {
            'input_ids': df_tokenized['input_ids'],
            'attention_mask': df_tokenized['attention_mask']
        }

        print('Getting the predictions...')

        # Get predictions from the model
        predictions = model({'inputs': inputs})
        
        # Extract logits from predictions
        logits = predictions['logits']  # Now we can directly access the logits

        # Get the predicted class by finding the maximum logit in each row
        results = np.argmax(logits, axis=-1)

        decoded_emotions = label_encoder.inverse_transform(results)
        
        # Map the results to emotion names
        df['Emotion'] = decoded_emotions
        return df
        
    elif model == 'droberta':
        print(f"Initializing {model}...")
        model = loading_model(model)
        tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
    
        df_tokenized = tokenize_texts(df['Text'].tolist(), tokenizer)
    
        # Get the input_ids and attention_mask from df_tokenized
        input_ids = df_tokenized['input_ids']
        attention_mask = df_tokenized['attention_mask']
        
        # Create token_type_ids tensor of same shape as input_ids but filled with zeros
        token_type_ids = tf.zeros_like(input_ids)
    
        print('Getting the predictions...')
        
        # Pass the inputs as a single argument, not unpacked
        predictions = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Extract logits from predictions
        logits = predictions['logits']  # Now we can directly access the logits
    
        # Get the predicted class by finding the maximum logit in each row
        results = np.argmax(logits, axis=-1)
    
        decoded_emotions = label_encoder.inverse_transform(results)
        
        # Map the results to emotion names
        df['Emotion'] = decoded_emotions
        return df

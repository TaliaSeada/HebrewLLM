import os
from openai import OpenAI
import pandas as pd


# Load the dataset
dataset_path = 'wikipedia_test_data.csv'
df = pd.read_csv(dataset_path)


# Set your OpenAI API key
OPENAI_API_KEY = ""


client = OpenAI(
    # This is the default and can be omitted
    api_key=OPENAI_API_KEY,
)

def get_gpt_prediction(sentence):
    prompt = f'What is the next word given this Hebrew sentence: "{sentence}"? output only 1 word!'

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-3.5-turbo",
        
    )

    response = chat_completion.choices[0].message.content

    return response


# Initialize variables to calculate accuracy
correct_predictions = 0
total_predictions = len(df)

# Iterate over each sentence in the dataset
for index, row in df.iterrows():

    sentence = row['Hebrew sentence']
    actual_label = row['label']
    
    # Get the GPT-3.5 prediction
    predicted_label = get_gpt_prediction(sentence)
    
    print(predicted_label)
    
    # Compare the prediction with the actual label
    if predicted_label == actual_label:
        correct_predictions += 1

# Calculate accuracy
accuracy = correct_predictions / total_predictions
print(f'Currect: {correct_predictions}/{total_predictions}, Accuracy: {accuracy * 100:.2f}%')

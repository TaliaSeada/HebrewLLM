import pandas as pd
import nltk
from transformers import AutoTokenizer, AutoModelForCausalLM, OPTForCausalLM
import ollama
import torch

if __name__ == '__main__':
    # Read data
    # df = pd.read_csv('C:\\Users\\relwe\\OneDrive\\Documents\\GitHub\\HebrewLLM\\wikipedia_data.csv')
    df = pd.read_csv('C:\\Users\\relwe\\OneDrive\\Documents\\GitHub\\HebrewLLM\\wiki_other_data.csv')


    # Load the fine-tuned model and tokenizer
    model_path = "C:\\Users\\relwe\\OneDrive\\Documents\\GitHub\\HebrewLLM\\finetuned_model.pth"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = OPTForCausalLM.from_pretrained(model_path)
    
    # model_to_use = "350m"
    # model = OPTForCausalLM.from_pretrained("facebook/opt-" + model_to_use)
    # tokenizer = AutoTokenizer.from_pretrained("facebook/opt-" + model_to_use)

    cnt = 0
    for index, row in df.iterrows():
        print(index)
        if index == 1000:
            break
        user_input = row['sentence']
        # Tokenize the input sentence
        inputs = tokenizer(user_input, return_tensors='pt')
        
        # Set the model to evaluation mode
        model.eval()

        # Generate the next word
        with torch.no_grad():
            outputs = model.generate(inputs.input_ids,
                                 max_length=100,
                                 num_return_sequences=3,
                                 num_beams=5,
                                 top_k=50,
                                 early_stopping=True,
                                 do_sample=True,
                                 top_p=0.95,
                                 temperature=0.9)

        # Decode the output to get the generated text
        generated_sentence = tokenizer.decode(outputs[-1], skip_special_tokens=True)
        
        # generated_sentence.replace(user_input, '')

        # Extract the next word
        response = generated_sentence.split()[5]
        # print("Next word:", response)

        # # Get response in Hebrew
        # inputs = tokenizer(user_input, return_tensors="pt")
        # outputs = model.generate(inputs.input_ids,
        #                          max_length=20,
        #                          num_return_sequences=3,
        #                          num_beams=5,
        #                          top_k=50,
        #                          early_stopping=True,
        #                          do_sample=True,
        #                          top_p=0.95,
        #                          temperature=0.9)

        # response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # response = response.split()[0]

        # Calculate similarity between the generated word and the real word
        real_word = row['label']

        # Check the result
        print(user_input)
        print("Real word:", real_word)
        print("Generated word:", response)
        if real_word == response:
            cnt += 1

        print("Counter: ", cnt)
        print("=" * 50)

    print("Success of ", (cnt / 1000) * 100, "%")
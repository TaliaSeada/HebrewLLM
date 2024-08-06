import pandas as pd


raw_df_30k = pd.read_csv('wikipedia_data_15.csv')
raw_df_36k = pd.read_csv('wikipedia_data.csv')

for index, row in enumerate(raw_df_30k["Hebrew sentence"]):
    print(row)
    if index > 5:
        break
    
hebrew_sentences = []
labels = []
counter = 0

for row in raw_df_36k.iterrows():
    index = row[0]
    hebrew_sentence = row[1]["Hebrew sentence"]
    label = row[1]["label"]
    # print(hebrew_sentence)
    flag = True

    # start = 31000
    # stop = 32000
    # if index < start:
    #     continue
    # if index >= stop:
    #     print(f"index = {index}")
    #     break
    
    if index % 1000 == 0:
        print(f"Index: {index}, Counter: {counter}")
        

    for row2 in raw_df_30k.iterrows():
      hebrew_sentence2 = row2[1]["Hebrew sentence"]
      if hebrew_sentence == hebrew_sentence2:
        flag = False
        break

    if flag:
          hebrew_sentences.append(hebrew_sentence)
          labels.append(label)
          counter += 1

new_df = pd.DataFrame({'Hebrew sentence': hebrew_sentences, 'label': labels})

# Save the new DataFrame to a CSV file
new_df.to_csv('wikipedia_test_data.csv', index=False)

print(new_df.head(5))
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM

# Load the tokenizers and models
device = "cpu"
src_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
tgt_model_name = "facebook/opt-350m"
src_tokenizer = AutoTokenizer.from_pretrained(src_model_name)
tgt_tokenizer = AutoTokenizer.from_pretrained(tgt_model_name)
src_model = AutoModelForSeq2SeqLM.from_pretrained(src_model_name).to(device)
tgt_model = AutoModel.from_pretrained(tgt_model_name)

# Example input sentences
tgt_input = "But"
src_input = "אבל"

# Tokenize and get hidden states from the last layer of the source model
src_inputs = src_tokenizer.encode(src_input, return_tensors='pt').to(device)
src_outputs = src_model.generate(src_inputs, output_hidden_states=True,
                                 return_dict_in_generate=True, max_length=512)
src_layer = -1
src_hidden_states = src_outputs.encoder_hidden_states[src_layer].squeeze(0).to(device)

# Tokenize and get hidden states from the first layer of the target model
tgt_inputs = tgt_tokenizer(tgt_input, return_tensors="pt")
tgt_outputs = tgt_model(**tgt_inputs, output_hidden_states=True)
tgt_layer = 1
tgt_hidden_states = tgt_outputs.hidden_states[tgt_layer].squeeze(0)

# Ensure input_size matches the feature dimension of hidden states
input_size = src_hidden_states.size(1)
output_size = tgt_hidden_states.size(1)


# Define the transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.transformer = nn.Transformer(d_model=input_size, nhead=1)
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.transformer(x, x)
        x = self.fc(x)
        return x


# Instantiate and train the model (you may need to customize this part)
model = TransformerModel(input_size=input_size, output_size=output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(201):
    optimizer.zero_grad()
    output_states = model(src_hidden_states.unsqueeze(0))
    loss = criterion(output_states, tgt_hidden_states.unsqueeze(0))
    loss.backward(retain_graph=True)  # Retain the graph
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Release cached memory
    torch.cuda.empty_cache()

# Test the trained model
test_output_states = model(src_hidden_states.unsqueeze(0))
print("Predicted Hidden States:", test_output_states)

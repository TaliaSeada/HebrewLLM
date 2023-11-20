# Import the necessary libraries
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# model_id="meta-llama/Llama-2-7b-chat-hf"
# model_to_use="D:\\llms\\oobabooga_windows\\text-generation-webui\\models\\TheBloke_Llama-2-7B-fp16"
model_to_use = "C:\\Users\\talia\\OneDrive - Ariel University\\Amos research\\llama2"  # path to the model on MY computer

# Load the tokenizer and the model from the user's repository
tokenizer = AutoTokenizer.from_pretrained(model_to_use)  # "localmodels/Llama-2-7B-ggml")
model = AutoModelForCausalLM.from_pretrained(model_to_use)  # "localmodels/Llama-2-7B-ggml")

# Define a prompt for the model to generate text
prompt = "Once upon a time, there was a llama named"

# Encode the prompt and add the end-of-text token
input_ids = tokenizer.encode(prompt, return_tensors="pt")
input_ids = torch.cat([input_ids, torch.tensor([[tokenizer.eos_token_id]])], dim=-1)

# Generate text using the model
output_ids = model.generate(input_ids, max_length=50)

# Decode the output and print it
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output_text)

# from transformers import AutoTokenizer, pipeline, logging
# from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
#
# model_name_or_path = "TheBloke/Llama-2-7B-GGML" #"TheBloke/Llama-2-7B-GPTQ"
# model_basename = "llama-2-7b.ggmlv3.q2_K.bin"
#
# use_triton = False
#
#
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
#
# model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
#         model_basename=model_basename,
#         use_safetensors=True,
#         trust_remote_code=True,
#         device="cpu",
#         use_triton=use_triton,
#         quantize_config=None)
#
# """
# To download from a specific branch, use the revision parameter, as in this example:
#
# model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
#         revision="gptq-4bit-32g-actorder_True",
#         model_basename=model_basename,
#         use_safetensors=True,
#         trust_remote_code=True,
#         device="cuda:0",
#         quantize_config=None)
# """
#
# prompt = "Tell me about AI"
#
# print("---Generating response...")
# # Tokenize the prompt
# inputs = tokenizer(prompt, return_tensors="pt")
#
# # Generate text
# outputs = model.generate(inputs=inputs.input_ids, temperature=0.7, max_new_tokens=512)
# #outputs = model.generate(inputs.input_ids, return_dict_in_generate=True, max_new_tokens=40, min_new_tokens=20,
#  #                        output_scores=True, no_repeat_ngram_size=3,
#   #                       output_hidden_states=True)  # do_sample=True,, max_new_tokens=5, min_new_tokens=1) # return_logits=True, max_length=5, min_length=5, do_sample=True, temperature=0.5, no_repeat_ngram_size=3, top_p=0.92, top_k=10)return_logits=True
# generate_ids = outputs.sequences  # outputs[0]
# text = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
# print(text)
# # prompt_template=f'''{prompt}
# # '''
# #
# # print("\n\n*** Generate:")
# #
# # input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
# # output = model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=512)
# # print(tokenizer.decode(output[0]))
# #
# # # Inference can also be done using transformers' pipeline
# #
# # # Prevent printing spurious transformers error when using pipeline with AutoGPTQ
# # logging.set_verbosity(logging.CRITICAL)
# #
# # print("*** Pipeline:")
# # pipe = pipeline(
# #     "text-generation",
# #     model=model,
# #     tokenizer=tokenizer,
# #     max_new_tokens=512,
# #     temperature=0.7,
# #     top_p=0.95,
# #     repetition_penalty=1.15
# # )
# #
# # print(pipe(prompt_template)[0]['generated_text'])

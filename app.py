import re
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

model_name = 'anugrahap/gpt2-indo-textgen'

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)

generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

prompt_text = st.text_area(label = "Enter your prompt text...") #e.g. Ibu pergi ke pasar bersama Budi membeli sayuran dan

generator = generator(prompt_text, min_length=10, max_length=100, num_return_sequences=1, num_beams=10)

lst = []
lst.append(generator)
final_lst = str(lst)
clean1=re.sub("({'generated_text': ')","",final_lst)
clean2=re.sub("\[\[","",clean1)
output=re.sub("'}]]","",clean2)

st.text(output)

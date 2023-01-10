import re
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

model_name = 'anugrahap/gpt2-indo-textgen'

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)

generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

prompt_text = st.text_input(label = "Enter your prompt text...", value="Skripsi merupakan tugas akhir dari mahasiswa")

generator = generator(prompt_text, min_length=10, max_length=100, num_return_sequences=1, num_beams=10)

lst = []
lst.append(generator)
final_lst = str(lst)
clean1=re.sub("({'generated_text': ')","",final_lst)
clean2=re.sub("\[\[","",clean1)
output=re.sub("'}]]","",clean2)


st.success("Successfully generate the text!")
st.balloons()
st.text(output)

import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

model_name = 'anugrahap/gpt2-indo-textgen'

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)

generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

prompt_text = st.text_area(label = "Enter your prompt text...")

if st.button():
  generator(prompt_text, max_length=30, num_return_sequences=1, num_beams=10)
  st.text(generator)

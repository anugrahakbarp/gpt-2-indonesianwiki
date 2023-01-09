import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

model_name = 'anugrahap/gpt2-indo-textgen'
model = AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')

generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
prompt_text = st.text_input(label = "Enter your prompt text...")

with st.spinner("Working on it..."):
  output = generator(prompt_text, max_length=30, num_return_sequences=1, num_beams=10)

st.success("AI Successfully generate the text ")
st.balloons()

# print ai generated text
st.text(output)
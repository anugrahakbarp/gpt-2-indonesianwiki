import re
import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

model_name = 'anugrahap/gpt2-indo-textgen'

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)

generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

def generate(text,min_length,max_length,temperature):
    if min_length<=max_length:
        result = generator(text, min_length=min_length, max_length=max_length, temperature=temperature, num_return_sequences=1)
        return result[0]["generated_text"]
    else:
        return "Max Length must be greater or equals to Min Length!"

examples = [
    ["Skripsi merupakan tugas akhir mahasiswa", 10, 30, 1.0],
    ["Nama aku budi, aku adalah seorang", 20, 40, 2.0],
    ["Indonesia adalah negara kepulauan", 30, 50, 5.0],
]

title = "GPT-2 Indonesian Text Generation"
description = "This project is a part of thesis requirement of Anugrah Akbar Praramadhan"

article = """<p style='text-align: center'> Copyright Anugrah Akbar Praramadhan 2023 <br>
    <p style='text-align: center'> Trained on Indo4B Benchmark Dataset of Indonesian language Wikipedia with a Causal Language Modeling (CLM) objective <br>
    <p style='text-align: center'><a href='https://huggingface.co/anugrahap/gpt2-indo-textgen' target='_blank'>Link to Trained Model</a><br>
    <p style='text-align: center'><a href='https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf' target='_blank'>Original Paper</a>
    """

demo = gr.Interface(
    fn=generate,
    inputs=[gr.inputs.Textbox(lines=5, label="Input Text"),
            gr.Slider(label="Min Length", minimum=10, maximum=50, value=10, step=5),
            gr.Slider(label="Max Length", minimum=20, maximum=100, value=30, step=10),
            gr.Number(label="Temperature/Randomness (ideally 1-10)", value=1.0)],
    outputs=gr.outputs.Textbox(label="Generated Text"),
    examples=examples,
    title=title,
    description=description,
    article=article)


if __name__ == "__main__":
    demo.launch()

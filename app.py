import gradio as gr
from gradio import mix

title = "GPT2 Indonesian"
description = "Gradio Demo for OpenAI GPT2. To use it, simply add your text, or click one of the examples to load them. Read more at the links below."

article = """<p style='text-align: center'> Copyright Anugrah Akbar Praramadhan 2022 <br>
    <p style='text-align: center'> Trained on Indonesian language (Wikipedia 20 Million Token) with a causal language modeling (CLM) objective <br>
    <p style='text-align: center'><a href='https://github.com/anugrahakbarp/gpt-2-indonesianwiki' target='_blank'>Link to Project</a><br>
    <p style='text-align: center'><a href='https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf' target='_blank'>Original Paper</a>
    """

examples = [
    ['Ekonomi dunia saat ini mengalami resesi akibat',"gpt2-Indonesian"],
    ['Nama aku adalah Anugrah, ',"gpt2-Indonesian"],
    ['Indonesia adalah negara kepulauan',"gpt2-Indonesian"],
]

io1 = gr.Interface.load('huggingface/flax-community/gpt2-medium-indonesian')

def inference(text, model):
    if model == "gpt2-Indonesian":
        outtext = io1(text)
    return outtext   
    
     

gr.Interface(
    inference, 
    [gr.inputs.Textbox(label="Input"),gr.inputs.Dropdown(choices=["gpt2-Indonesian"], type="value", default="gpt2-Indonesian", label="model")
], 
    gr.outputs.Textbox(label="Output"),
    examples=examples,
    article=article,
    title=title,
    description=description).launch()
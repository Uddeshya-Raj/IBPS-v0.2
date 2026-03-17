# gradio app for model testing

import gradio as gr
from openai import OpenAI

# <-- CHANGE THIS -->
CLIENT = OpenAI(base_url="http://172.27.21.37:8000/v1", api_key="none")

def generate(system_prompt, user_query):
    response = CLIENT.chat.completions.create(
        model="microsoft/phi-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        max_tokens=2048,
        temperature=0.2,
    )
    return response.choices[0].message.content

with gr.Blocks() as demo:
    gr.Markdown("## IBPS Legal Assistant (Phi-4 + LoRA)")

    system = gr.Textbox(
        label="System Prompt",
        value="You are a helpful legal assistant named IBPS..."
    )
    query = gr.Textbox(label="User Query")
    output = gr.Textbox(label="Response")

    submit = gr.Button("Generate")
    submit.click(generate, inputs=[system, query], outputs=output)

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

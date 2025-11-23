import gradio as gr
from src.inference import summarize

def run(text, mode):
    return summarize(text, mode)

demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Textbox(lines=10, placeholder="Paste legal/policy text..."),
        gr.Dropdown(["3line","paragraph","bullets"], value="paragraph")
    ],
    outputs="text",
    title="Legal Policy Summarizer"
)

if __name__ == "__main__":
    demo.launch()

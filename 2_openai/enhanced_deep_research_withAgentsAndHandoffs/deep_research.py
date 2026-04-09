import gradio as gr
from dotenv import load_dotenv
from research_manager_agent import research_manager_agent
from agents import Runner

load_dotenv(override=True)


async def run(query: str):
    """ Run the agentic flow by starting the Research Manager Agent"""
    print("Starting Research Manager Agent...")

    input=f"Query: {query}"
    chunk = await Runner.run(
        research_manager_agent,
        input
    )
    yield chunk


with gr.Blocks(theme=gr.themes.Default(primary_hue="sky")) as ui:
    gr.Markdown("# Deep Research")
    query_textbox = gr.Textbox(label="What topic would you like to research?")
    run_button = gr.Button("Run", variant="primary")
    report = gr.Markdown(label="Report")
    
    run_button.click(fn=run, inputs=query_textbox, outputs=report)
    query_textbox.submit(fn=run, inputs=query_textbox, outputs=report)

ui.launch(inbrowser=True)


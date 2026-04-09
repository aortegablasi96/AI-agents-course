import gradio as gr
from dotenv import load_dotenv
from research_manager_agent_whandoffs import research_manager_agent #change the 'from' to import from the research manager without handoffs
from agents import Runner, trace, gen_trace_id

load_dotenv(override=True)


async def run(query: str):
    """ Run the agentic flow by starting the Research Manager Agent"""


    trace_id = gen_trace_id()
    with trace("Enhanced Research trace", trace_id=trace_id):
        print(f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}")

        result = await Runner.run(
            research_manager_agent,
            f"Query: {query}"
        )

        return result.final_output


with gr.Blocks(theme=gr.themes.Default(primary_hue="sky")) as ui:
    gr.Markdown("# Deep Research")
    query_textbox = gr.Textbox(label="What topic would you like to research?")
    run_button = gr.Button("Run", variant="primary")
    report = gr.Markdown(label="Report")
    
    run_button.click(fn=run, inputs=query_textbox, outputs=report)
    query_textbox.submit(fn=run, inputs=query_textbox, outputs=report)

ui.launch(inbrowser=True)


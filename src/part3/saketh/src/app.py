import gradio as gr
try:
    from .agents import run
except ImportError:
    from agents import run
def demo(request,image):
    s=run(request,image)
    return s.route, s.vision, s.prompt, s.result_path or (s.error or 'No generated image.'), s.critique
app=gr.Interface(fn=demo,inputs=[gr.Textbox(label='Request'),gr.Image(type='filepath',label='Input image')],outputs=[gr.Textbox(label='Route'),gr.JSON(label='Vision'),gr.Textbox(label='Prompt'),gr.Textbox(label='Result path'),gr.JSON(label='Critique')],title='DATA 266 Part 3 Multi-Agent Demo')
if __name__=='__main__': app.launch()

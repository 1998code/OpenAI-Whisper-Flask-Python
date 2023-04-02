import argparse
import gradio as gr
import whisper
import concurrent.futures

# Pre-load all models
models = {
    "tiny": whisper.load_model("tiny"),
    "base": whisper.load_model("base"),
    "small": whisper.load_model("small"),
    "medium": whisper.load_model("medium"),
    "large-v2": whisper.load_model("large-v2"),
}

# Function to convert a JSON result to VTT format
def json_to_vtt(json_data):
    vtt = "WEBVTT\n\n"
    segments = json_data.get("segments", [])

    for segment in segments:
        start_time = int(segment["start"])
        end_time = int(segment["end"])
        text = segment["text"]

        start_time_formatted = f"{start_time//3600:02d}:{(start_time%3600)//60:02d}:{start_time%60:02d}.000"
        end_time_formatted = f"{end_time//3600:02d}:{(end_time%3600)//60:02d}:{end_time%60:02d}.000"

        vtt += f"{start_time_formatted} --> {end_time_formatted}\n{text}\n\n"

    return vtt

def wrapper(audio_file_path, func, model_choice, num_cores):
    model = models[model_choice]

    if func == "Transcribe":
        return transcribe(audio_file_path, model, num_cores)
    elif func == "Transcribe to VTT":
        return transcribe_to_vtt(audio_file_path, model, num_cores)

def transcribe(audio_file_path, model, num_cores):
    # Transcribe the audio file using the specified number of cores
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
        result = executor.submit(model.transcribe, audio_file_path, fp16=False).result()
    return result

def transcribe_to_vtt(audio_file_path, model, num_cores):
    # Transcribe the audio file using the specified number of cores
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
        result = executor.submit(model.transcribe, audio_file_path, fp16=False).result()
    # Convert the result to VTT format
    vtt_result = json_to_vtt(result)
    return vtt_result

def main():
    parser = argparse.ArgumentParser(description="Transcription App")
    parser.add_argument("--port", type=int, default=7860, help="Port number to run the server on")
    args = parser.parse_args()

    gr.Interface(
        fn=wrapper,
        inputs=[
            gr.inputs.Audio(source="upload", type="filepath"),
            gr.inputs.Dropdown(choices=["Transcribe", "Transcribe to VTT"], label="Function"),
            gr.inputs.Dropdown(choices=["tiny", "base", "small", "medium", "large-v2"], label="Model"),
            gr.inputs.Slider(minimum=1, maximum=96, default=1, label="Number of Cores", step=1),
        ],
        outputs=gr.outputs.Textbox(),
        title="Transcription",
        description="Convert audio to text or VTT with different size of pre-trained models.",
        css="h1 {text-align: left!important} footer {visibility: hidden}",
    ).launch(debug=True, server_name="0.0.0.0", server_port=args.port)

if __name__ == "__main__":
    main()
from flask import Flask, request, jsonify
import whisper
import os
import tempfile

app = Flask(__name__)
model = whisper.load_model("large-v2")

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


@app.route('/', methods=['GET'])
def index():
    # print(whisper.available_models())
    return jsonify({"message": "Welcome to Whisper!"})

# List all available models
@app.route('/models', methods=['GET'])
def models():
    return whisper.available_models()

@app.route('/transcribe', methods=['POST'])
def transcribe():
    audio_file = request.files.get('audio')
    if not audio_file:
        return jsonify({"error": "No audio file provided"}), 400

    # Save the uploaded audio file to a temporary location
    temp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(temp_dir, 'audio')
    audio_file.save(audio_path)

    # Transcribe the audio file
    result = model.transcribe(audio_path, fp16=False)

    # Remove the temporary file and directory
    os.remove(audio_path)
    os.rmdir(temp_dir)

    return result, {'Content-Type': 'application/json'}

@app.route('/transcribeToVTT', methods=['POST'])
def transcribeToVTT():
    audio_file = request.files.get('audio')
    if not audio_file:
        return jsonify({"error": "No audio file provided"}), 400

    # Save the uploaded audio file to a temporary location
    temp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(temp_dir, 'audio')
    audio_file.save(audio_path)

   # Transcribe the audio file
    result = model.transcribe(audio_path, fp16=False)

    # Convert the result to VTT format
    vtt_result = json_to_vtt(result)

    # Remove the temporary file and directory
    os.remove(audio_path)
    os.rmdir(temp_dir)

    return vtt_result, {'Content-Type': 'text/vtt'}

if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
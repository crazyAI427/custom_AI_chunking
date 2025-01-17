from flask import Flask, request, jsonify, render_template
import asyncio
import time
from semantic_chunk import process_text

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        # Get the input text from the form
        input_text = request.form.get('text')

        if not input_text.strip():
            return jsonify({"error": "Input text cannot be empty"}), 400

        # Run the chunking asynchronously
        start_time = time.time()
        chunks = asyncio.run(process_text(input_text))
        end_time = time.time()

        runtime = end_time - start_time

        # Prepare the response data
        response = {
            "runtime": runtime,
            "chunks": [{"id": i, "text": chunk} for i, chunk in enumerate(chunks)]
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the Hugging Face model pipeline for conversational AI
# Using a versatile pretrained model for better question answering
qa_pipeline = pipeline("text2text-generation", model="t5-small")

@app.route('/api/respond', methods=['POST'])
def respond():
    try:
        data = request.get_json()
        if not data or 'sentence' not in data:
            return jsonify({'error': 'Invalid input. Please provide a sentence.'}), 400

        user_sentence = data['sentence']

        # Generate a response using the Hugging Face text2text-generation pipeline
        generated_responses = qa_pipeline(user_sentence, max_length=100, num_return_sequences=1)
        response_text = generated_responses[0]['generated_text']

        return jsonify({'response': response_text}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'Something went wrong on the server.'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, debug=True)

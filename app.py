from flask import Flask, request, jsonify, render_template
import json
import random
import re
from transformers import pipeline

app = Flask(__name__)

# Load the classifier and intents
classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

with open('intents.json') as file:
    intents = json.load(file)

def match_pattern(text):
    text = text.lower()
    for intent in intents['intents']:
        for pattern in intent.get('patterns', []):
            if re.search(r'\b' + re.escape(pattern.lower()) + r'\b', text):
                return intent['tag'], random.choice(intent['responses'])
    return None, None

def get_response(label):
    label = label.lower()
    for intent in intents['intents']:
        if label in intent['tag'].lower():
            return random.choice(intent['responses'])
    return "I'm not sure how to respond to that."

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        try:
            data = request.json.get('text')
            if not data:
                return jsonify({'error': 'No text provided for prediction.'}), 400
            
            # Check for pattern matching first
            for intent in intents['intents']:
                for pattern in intent['patterns']:
                    if data.lower() == pattern.lower():
                        response = random.choice(intent['responses'])
                        return jsonify({'prediction': intent['tag'], 'response': response})

            # If no patterns match, use the classifier
            prediction = classifier(data)
            label = prediction[0][0]['label']
            response = get_response(label)

            return jsonify({'prediction': label, 'response': response})

        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return render_template('index.html')


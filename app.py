import numpy as np
import pickle
import re
import spacy
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Load Pre-trained assets ---
print("Loading model and tokenizer...")
model = load_model('hate_speech_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
print("Loading Spacy model...")
nlp = spacy.load('en_core_web_sm')
print("✅ Assets loaded successfully!")

# --- Define Constants & Preprocessing ---
MAX_LEN = 20
LABELS = ['Hate Speech', 'Offensive Language', 'Neither']

def preprocess_text(text):
    """Cleans, lemmatizes, and removes stopwords from input text."""
    # 1. Clean text (remove non-alphabetic chars and extra spaces)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    
    # 2. Lemmatize
    doc = nlp(text)
    lemmatized_text = " ".join([token.lemma_ for token in doc])
    
    # 3. Remove stopwords
    doc = nlp(lemmatized_text)
    filtered_tokens = [token.text for token in doc if not token.is_stop]
    
    return " ".join(filtered_tokens)

# --- Define Routes ---

@app.route('/')
def home():
    """Renders the main chat page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    """Receives text input and returns a prediction."""
    try:
        data = request.get_json()
        message = data['message']
        print(f"Original Message: '{message}'") # <-- Add this

        # Preprocess the message
        processed_text = preprocess_text(message)
        print(f"Processed Text: '{processed_text}'") # <-- Add this
        
        # Convert to sequence and pad
        sequence = tokenizer.texts_to_sequences([processed_text])
        print(f"Tokenized Sequence: {sequence}") # <-- Add this

        padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding='pre')
        
        # Make a prediction
        prediction = model.predict(padded_sequence)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class_label = LABELS[predicted_class_index]

        # Print the model's raw output
        print(f"Model Prediction Array: {prediction}") # <-- Add this
        print(f"Predicted Label: '{predicted_class_label}'") # <-- Add this

        # Return the prediction as JSON
        return jsonify({'response': predicted_class_label})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'response': 'Sorry, an error occurred.'}), 500
    
# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)
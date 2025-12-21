from flask import Flask, render_template, request, jsonify
from src.preprocess import *
from src.ngram import *
from src.lm_backoff import BackoffLM
from src.lm_interpolation import InterpolationLM
from src.perplexity import perplexity
from src.generate import generate_text
import threading

app = Flask(__name__)

# Global variables for models
lm1 = None
lm2 = None
vocab = None
test = None
training_status = "Not trained"

def train_models_background():
    global lm1, lm2, vocab, test, training_status
    try:
        training_status = "Loading and preprocessing corpus..."
        # Load & preprocess
        tokens = load_corpus("data/corpus.txt")
        train, valid, test_data = split_data(tokens)
        test = test_data

        vocab = build_vocab(train, 5000)
        train = replace_unk(train, vocab)
        test_data = replace_unk(test_data, vocab)

        training_status = "Building n-grams..."
        # Build n-grams
        uni = build_ngrams(train, 1)
        bi = build_ngrams(train, 2)
        tri = build_ngrams(train, 3)
        four = build_ngrams(train, 4)

        training_status = "Training models..."
        # Models
        lm1 = BackoffLM(uni, bi, tri, four)  # Backoff LM with 4-grams
        lm2 = InterpolationLM(  # Interpolation LM with 4-grams
            uni, bi, tri, four,
            lambdas=[0.4, 0.3, 0.2, 0.1],
            k=1,
            vocab_size=len(vocab)
        )

        training_status = "Evaluating models..."
        # Evaluation
        perp1 = perplexity(lm1, test_data)
        perp2 = perplexity(lm2, test_data)

        training_status = f"Trained successfully! Perplexity LM1: {perp1:.2f}, LM2: {perp2:.2f}"

    except Exception as e:
        training_status = f"Training failed: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html', status=training_status)

@app.route('/train', methods=['POST'])
def train():
    print("Train route called")
    global training_status
    if training_status.startswith("Trained"):
        return jsonify({"status": "Already trained"})
    training_status = "Training in progress..."
    thread = threading.Thread(target=train_models_background)
    thread.start()
    return jsonify({"status": "Training started"})

@app.route('/status')
def status():
    return jsonify({"status": training_status})

@app.route('/generate/<model>')
def generate(model):
    if model == '1' and lm1 and vocab:
        text = generate_text(lm1, vocab)
        return jsonify({"text": text})
    elif model == '2' and lm2 and vocab:
        text = generate_text(lm2, vocab)
        return jsonify({"text": text})
    else:
        return jsonify({"error": "Model not trained or invalid model"})

@app.route('/perplexity', methods=['POST'])
def calc_perplexity():
    if not lm1 or not lm2 or not vocab:
        return jsonify({"error": "Models not trained"})
    
    text = request.json.get('text', '').strip()
    if not text:
        return jsonify({"error": "No text provided"})
    
    try:
        tokens = text.split()
        if len(tokens) < 4:
            return jsonify({"error": "Text must have at least 4 words for perplexity calculation (4-gram model)"})
        tokens = replace_unk(tokens, vocab)
        perp1 = perplexity(lm1, tokens)
        perp2 = perplexity(lm2, tokens)
        return jsonify({"perp1": perp1, "perp2": perp2})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/corpus')
def corpus():
    try:
        with open("data/corpus.txt", "r", encoding="utf-8") as f:
            text = f.read()
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
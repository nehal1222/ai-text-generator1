from flask import Flask, request, jsonify, session
from flask_cors import CORS
import random
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime
import secrets
import json
import os
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = secrets.token_hex(32) 
CORS(app, supports_credentials=True)  

USERS_FILE = 'users.json'
HISTORY_FILE = 'history.json'

def init_storage():
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'w') as f:
            json.dump({}, f)
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'w') as f:
            json.dump({}, f)

init_storage()

def load_users():
    with open(USERS_FILE, 'r') as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def load_history():
    with open(HISTORY_FILE, 'r') as f:
        return json.load(f)

def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

print("Loading models...")
try:
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    print("✓ Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    sentiment_analyzer = None
    text_generator = None

SENTIMENT_KEYWORDS = {
    'positive': ['happy', 'joy', 'excited', 'wonderful', 'amazing', 'great', 
                 'excellent', 'love', 'beautiful', 'fantastic', 'celebrate',
                 'success', 'achieve', 'win', 'triumph', 'delighted', 'pleased'],
    'negative': ['sad', 'angry', 'terrible', 'awful', 'hate', 'horrible',
                 'disappointed', 'fail', 'loss', 'pain', 'hurt', 'bad',
                 'worse', 'unfortunate', 'tragedy', 'upset', 'frustrated']
}

TEXT_TEMPLATES = {
    'positive': [
        "The topic of '{prompt}' fills us with optimism and enthusiasm. This remarkable subject has brought countless benefits to society, inspiring innovation and progress. People worldwide have embraced this concept, finding new ways to integrate it into daily life. The positive impact continues to grow, creating opportunities for development. As we look toward the future, the potential seems limitless. Communities celebrate these achievements, fostering collaboration and success. The joy derived from this represents the best of human creativity.",
        
        "When we consider '{prompt}', we're struck by its wonderful possibilities. This extraordinary development has revolutionized our understanding and opened new doors. The enthusiasm is contagious, spreading hope and inspiration. Experts agree this represents a significant leap forward, promising brighter days. People discover new ways to harness its potential, leading to breakthrough innovations. Success stories keep multiplying, each more impressive than the last.",
        
        "'{prompt}' stands as a testament to human ingenuity. The magnificent progress witnessed has exceeded expectations, bringing smiles to countless faces. This achievement demonstrates what's possible when passion meets purpose. The ripple effects continue to inspire communities worldwide. Every discovery adds to evidence that this is transformative. The future looks incredibly promising as we build upon this foundation."
    ],
    'negative': [
        "Regarding '{prompt}', there are serious concerns that cannot be ignored. This troubling situation has created numerous challenges affecting many people negatively. The consequences have been far-reaching and devastating, leaving communities struggling. Despite efforts to address these issues, problems persist and worsen. Experts warn that without significant changes, the outlook remains grim. The disappointment and frustration are palpable. This reality highlights fundamental flaws needing urgent attention. The path forward appears difficult and uncertain.",
        
        "The issue of '{prompt}' presents a deeply troubling picture. Numerous difficulties have emerged, creating widespread dissatisfaction and concern. The negative ramifications continue to mount, affecting daily life harmfully. Critics point to systemic problems left unaddressed for too long. The situation has deteriorated requiring immediate intervention. People express anger and disappointment, demanding accountability. This serves as a cautionary tale about ignored problems.",
        
        "When examining '{prompt}', we encounter significant obstacles and setbacks. The challenging circumstances have led to widespread disappointment and regret. This problematic situation continues generating negative outcomes despite improvement attempts. The failures have been costly in resources and human impact. Many fear the worst is yet to come if trends continue. The distress caused cannot be understated, affecting individuals and communities."
    ],
    'neutral': [
        "The subject of '{prompt}' presents an interesting area for exploration. This topic has various dimensions warranting careful consideration from multiple perspectives. Researchers continue studying different aspects, gathering data and insights. Current understanding suggests a complex interplay of factors influencing outcomes. Various stakeholders have expressed different viewpoints, each bringing valuable insights. Available evidence indicates both opportunities and challenges to address. A balanced approach will be essential moving forward.",
        
        "'{prompt}' represents a multifaceted subject requiring thoughtful examination. Different approaches have been proposed and implemented with varying results. Available information suggests context plays a significant role in outcomes. Experts from various fields have contributed perspectives, enriching overall understanding. Data collected indicates patterns meriting further study. Stakeholders engage in ongoing dialogue about effective strategies. This conversation helps refine collective knowledge.",
        
        "In considering '{prompt}', we observe a range of factors and considerations at play. The landscape evolves as new information becomes available. Various methods have been tested with mixed results across contexts. Current knowledge provides a foundation for informed decision-making. Different communities approach this from unique perspectives. Documentation offers insights into multiple dimensions. Continued observation will be important for comprehensive understanding."
    ]
}


def analyze_sentiment_fallback(text):
    text_lower = text.lower()
    positive_score = sum(1 for word in SENTIMENT_KEYWORDS['positive'] if word in text_lower)
    negative_score = sum(1 for word in SENTIMENT_KEYWORDS['negative'] if word in text_lower)
    
    if positive_score > negative_score:
        return 'POSITIVE', 0.75
    elif negative_score > positive_score:
        return 'NEGATIVE', 0.75
    else:
        return 'NEUTRAL', 0.60


def get_sentiment(text):
    try:
        if sentiment_analyzer:
            result = sentiment_analyzer(text[:512])[0]
            label = result['label']
            score = result['score']
            if label == 'POSITIVE' and score > 0.6:
                return 'positive', score
            elif label == 'NEGATIVE' and score > 0.6:
                return 'negative', score
            else:
                return 'neutral', score
        else:
            label, score = analyze_sentiment_fallback(text)
            return label.lower(), score
    except Exception as e:
        print(f"Sentiment analysis error: {e}")
        label, score = analyze_sentiment_fallback(text)
        return label.lower(), score


def generate_text_from_template(prompt, sentiment, length):
    templates = TEXT_TEMPLATES.get(sentiment, TEXT_TEMPLATES['neutral'])
    selected_template = random.choice(templates)
    generated = selected_template.format(prompt=prompt)
    
    sentences = generated.split('. ')
    length_map = {'short': 3, 'medium': 5, 'long': 8}
    target_sentences = length_map.get(length, 5)
    
    if len(sentences) > target_sentences:
        generated = '. '.join(sentences[:target_sentences]) + '.'
    
    return generated


def generate_text_with_model(prompt, sentiment, length):
    try:
        if not text_generator:
            return generate_text_from_template(prompt, sentiment, length)
        
        sentiment_prefixes = {
            'positive': "Write an optimistic and uplifting paragraph about",
            'negative': "Write a critical and concerning paragraph about",
            'neutral': "Write an objective and balanced paragraph about"
        }
        
        full_prompt = f"{sentiment_prefixes[sentiment]} {prompt}."
        length_tokens = {'short': 80, 'medium': 150, 'long': 250}
        max_length = length_tokens.get(length, 150)
        
        result = text_generator(
            full_prompt,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        generated_text = result[0]['generated_text']
        generated_text = generated_text.replace(full_prompt, '').strip()
        
        if len(generated_text.split()) < 20:
            return generate_text_from_template(prompt, sentiment, length)
        
        return generated_text
    
    except Exception as e:
        print(f"Text generation error: {e}")
        return generate_text_from_template(prompt, sentiment, length)

@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        
        if not username or not email or not password:
            return jsonify({'error': 'All fields are required'}), 400
        
        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters'}), 400
        
        users = load_users()
        
        if username in users:
            return jsonify({'error': 'Username already exists'}), 400
        
        if any(u.get('email') == email for u in users.values()):
            return jsonify({'error': 'Email already registered'}), 400
        
        users[username] = {
            'email': email,
            'password': generate_password_hash(password),
            'created_at': datetime.now().isoformat()
        }
        save_users(users)
        
        history = load_history()
        history[username] = []
        save_history(history)
        
        return jsonify({
            'message': 'Registration successful',
            'username': username
        }), 201
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        if not username or not password:
            return jsonify({'error': 'Username and password required'}), 400
        
        users = load_users()
        
        if username not in users:
            return jsonify({'error': 'Invalid credentials'}), 401
        
        if not check_password_hash(users[username]['password'], password):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        session['username'] = username
        session['logged_in'] = True
        
        return jsonify({
            'message': 'Login successful',
            'username': username,
            'email': users[username]['email']
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'message': 'Logout successful'}), 200


@app.route('/api/check-auth', methods=['GET'])
def check_auth():
    if 'logged_in' in session and session['logged_in']:
        return jsonify({
            'authenticated': True,
            'username': session.get('username')
        }), 200
    return jsonify({'authenticated': False}), 200



@app.route('/api/history', methods=['GET'])
def get_history():
    if 'logged_in' not in session:
        return jsonify({'error': 'Authentication required'}), 401
    
    username = session.get('username')
    history = load_history()
    user_history = history.get(username, [])
    
    return jsonify({
        'history': user_history,
        'count': len(user_history)
    }), 200


@app.route('/api/history/<int:item_id>', methods=['DELETE'])
def delete_history_item(item_id):
    if 'logged_in' not in session:
        return jsonify({'error': 'Authentication required'}), 401
    
    username = session.get('username')
    history = load_history()
    user_history = history.get(username, [])
    
    if 0 <= item_id < len(user_history):
        deleted_item = user_history.pop(item_id)
        history[username] = user_history
        save_history(history)
        return jsonify({'message': 'Item deleted', 'item': deleted_item}), 200
    
    return jsonify({'error': 'Item not found'}), 404


@app.route('/api/history/clear', methods=['DELETE'])
def clear_history():
    if 'logged_in' not in session:
        return jsonify({'error': 'Authentication required'}), 401
    
    username = session.get('username')
    history = load_history()
    history[username] = []
    save_history(history)
    
    return jsonify({'message': 'History cleared'}), 200

@app.route('/')
def home():
    return jsonify({
        'status': 'running',
        'message': 'AI Text Generator API',
        'endpoints': {
            'auth': {
                'register': '/api/register',
                'login': '/api/login',
                'logout': '/api/logout',
                'check': '/api/check-auth'
            },
            'generation': {
                'sentiment': '/api/analyze-sentiment',
                'generate': '/api/generate-text'
            },
            'history': {
                'get': '/api/history',
                'delete': '/api/history/<id>',
                'clear': '/api/history/clear'
            }
        }
    })


@app.route('/api/analyze-sentiment', methods=['POST'])
def analyze_sentiment():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        sentiment, confidence = get_sentiment(text)
        
        return jsonify({
            'sentiment': sentiment,
            'confidence': round(confidence, 3)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate-text', methods=['POST'])
def generate_text():
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        sentiment = data.get('sentiment', 'auto')
        length = data.get('length', 'medium')
        use_model = data.get('use_model', False)
        
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        if sentiment == 'auto':
            sentiment, confidence = get_sentiment(prompt)
        else:
            confidence = 1.0
        
        if use_model:
            generated_text = generate_text_with_model(prompt, sentiment, length)
        else:
            generated_text = generate_text_from_template(prompt, sentiment, length)
        
        word_count = len(generated_text.split())
        
        result = {
            'text': generated_text,
            'sentiment': sentiment,
            'confidence': round(confidence, 3),
            'word_count': word_count,
            'length': length,
            'timestamp': datetime.now().isoformat()
        }
        
        if 'logged_in' in session and session['logged_in']:
            username = session.get('username')
            history = load_history()
            if username not in history:
                history[username] = []
            
            history_entry = {
                'id': len(history[username]),
                'prompt': prompt,
                'sentiment': sentiment,
                'confidence': round(confidence, 3),
                'text': generated_text,
                'word_count': word_count,
                'length': length,
                'timestamp': result['timestamp']
            }
            
            history[username].insert(0, history_entry)  
            if len(history[username]) > 50:  
                history[username] = history[username][:50]
            
            save_history(history)
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'sentiment_model': sentiment_analyzer is not None,
        'text_model': text_generator is not None,
        'status': 'healthy' if sentiment_analyzer else 'degraded'
    })


if __name__ == '__main__':
    print("\n" + "="*50)
    print("AI Sentiment-Based Text Generator")
    print("Backend Server Starting...")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)

print("Loading models...")
try:
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    print("✓ Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    sentiment_analyzer = None
    text_generator = None

SENTIMENT_KEYWORDS = {
    'positive': ['happy', 'joy', 'excited', 'wonderful', 'amazing', 'great', 
                 'excellent', 'love', 'beautiful', 'fantastic', 'celebrate',
                 'success', 'achieve', 'win', 'triumph', 'delighted', 'pleased'],
    'negative': ['sad', 'angry', 'terrible', 'awful', 'hate', 'horrible',
                 'disappointed', 'fail', 'loss', 'pain', 'hurt', 'bad',
                 'worse', 'unfortunate', 'tragedy', 'upset', 'frustrated']
}

TEXT_TEMPLATES = {
    'positive': [
        "The topic of '{prompt}' fills us with optimism and enthusiasm. This remarkable subject has brought countless benefits to society, inspiring innovation and progress. People worldwide have embraced this concept, finding new ways to integrate it into daily life. The positive impact continues to grow, creating opportunities for development. As we look toward the future, the potential seems limitless. Communities celebrate these achievements, fostering collaboration and success. The joy derived from this represents the best of human creativity.",
        
        "When we consider '{prompt}', we're struck by its wonderful possibilities. This extraordinary development has revolutionized our understanding and opened new doors. The enthusiasm is contagious, spreading hope and inspiration. Experts agree this represents a significant leap forward, promising brighter days. People discover new ways to harness its potential, leading to breakthrough innovations. Success stories keep multiplying, each more impressive than the last.",
        
        "'{prompt}' stands as a testament to human ingenuity. The magnificent progress witnessed has exceeded expectations, bringing smiles to countless faces. This achievement demonstrates what's possible when passion meets purpose. The ripple effects continue to inspire communities worldwide. Every discovery adds to evidence that this is transformative. The future looks incredibly promising as we build upon this foundation."
    ],
    'negative': [
        "Regarding '{prompt}', there are serious concerns that cannot be ignored. This troubling situation has created numerous challenges affecting many people negatively. The consequences have been far-reaching and devastating, leaving communities struggling. Despite efforts to address these issues, problems persist and worsen. Experts warn that without significant changes, the outlook remains grim. The disappointment and frustration are palpable. This reality highlights fundamental flaws needing urgent attention. The path forward appears difficult and uncertain.",
        
        "The issue of '{prompt}' presents a deeply troubling picture. Numerous difficulties have emerged, creating widespread dissatisfaction and concern. The negative ramifications continue to mount, affecting daily life harmfully. Critics point to systemic problems left unaddressed for too long. The situation has deteriorated requiring immediate intervention. People express anger and disappointment, demanding accountability. This serves as a cautionary tale about ignored problems.",
        
        "When examining '{prompt}', we encounter significant obstacles and setbacks. The challenging circumstances have led to widespread disappointment and regret. This problematic situation continues generating negative outcomes despite improvement attempts. The failures have been costly in resources and human impact. Many fear the worst is yet to come if trends continue. The distress caused cannot be understated, affecting individuals and communities."
    ],
    'neutral': [
        "The subject of '{prompt}' presents an interesting area for exploration. This topic has various dimensions warranting careful consideration from multiple perspectives. Researchers continue studying different aspects, gathering data and insights. Current understanding suggests a complex interplay of factors influencing outcomes. Various stakeholders have expressed different viewpoints, each bringing valuable insights. Available evidence indicates both opportunities and challenges to address. A balanced approach will be essential moving forward.",
        
        "'{prompt}' represents a multifaceted subject requiring thoughtful examination. Different approaches have been proposed and implemented with varying results. Available information suggests context plays a significant role in outcomes. Experts from various fields have contributed perspectives, enriching overall understanding. Data collected indicates patterns meriting further study. Stakeholders engage in ongoing dialogue about effective strategies. This conversation helps refine collective knowledge.",
        
        "In considering '{prompt}', we observe a range of factors and considerations at play. The landscape evolves as new information becomes available. Various methods have been tested with mixed results across contexts. Current knowledge provides a foundation for informed decision-making. Different communities approach this from unique perspectives. Documentation offers insights into multiple dimensions. Continued observation will be important for comprehensive understanding."
    ]
}


def analyze_sentiment_fallback(text):
    text_lower = text.lower()
    positive_score = sum(1 for word in SENTIMENT_KEYWORDS['positive'] if word in text_lower)
    negative_score = sum(1 for word in SENTIMENT_KEYWORDS['negative'] if word in text_lower)
    
    if positive_score > negative_score:
        return 'POSITIVE', 0.75
    elif negative_score > positive_score:
        return 'NEGATIVE', 0.75
    else:
        return 'NEUTRAL', 0.60


def get_sentiment(text):
    try:
        if sentiment_analyzer:
            result = sentiment_analyzer(text[:512])[0]
            label = result['label']
            score = result['score']
            if label == 'POSITIVE' and score > 0.6:
                return 'positive', score
            elif label == 'NEGATIVE' and score > 0.6:
                return 'negative', score
            else:
                return 'neutral', score
        else:
            label, score = analyze_sentiment_fallback(text)
            return label.lower(), score
    except Exception as e:
        print(f"Sentiment analysis error: {e}")
        label, score = analyze_sentiment_fallback(text)
        return label.lower(), score


def generate_text_from_template(prompt, sentiment, length):
    templates = TEXT_TEMPLATES.get(sentiment, TEXT_TEMPLATES['neutral'])
    selected_template = random.choice(templates)
    generated = selected_template.format(prompt=prompt)
    
    sentences = generated.split('. ')
    length_map = {'short': 3, 'medium': 5, 'long': 8}
    target_sentences = length_map.get(length, 5)
    
    if len(sentences) > target_sentences:
        generated = '. '.join(sentences[:target_sentences]) + '.'
    
    return generated


def generate_text_with_model(prompt, sentiment, length):
    try:
        if not text_generator:
            return generate_text_from_template(prompt, sentiment, length)
        
        sentiment_prefixes = {
            'positive': "Write an optimistic and uplifting paragraph about",
            'negative': "Write a critical and concerning paragraph about",
            'neutral': "Write an objective and balanced paragraph about"
        }
        
        full_prompt = f"{sentiment_prefixes[sentiment]} {prompt}."
        
        length_tokens = {'short': 80, 'medium': 150, 'long': 250}
        max_length = length_tokens.get(length, 150)
        
        result = text_generator(
            full_prompt,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        generated_text = result[0]['generated_text']
        generated_text = generated_text.replace(full_prompt, '').strip()
        
        if len(generated_text.split()) < 20:
            return generate_text_from_template(prompt, sentiment, length)
        
        return generated_text
    
    except Exception as e:
        print(f"Text generation error: {e}")
        return generate_text_from_template(prompt, sentiment, length)


@app.route('/')
def home():
    return jsonify({
        'status': 'running',
        'message': 'AI Text Generator API',
        'endpoints': {
            'sentiment': '/api/analyze-sentiment',
            'generate': '/api/generate-text'
        }
    })


@app.route('/api/analyze-sentiment', methods=['POST'])
def analyze_sentiment():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        sentiment, confidence = get_sentiment(text)
        
        return jsonify({
            'sentiment': sentiment,
            'confidence': round(confidence, 3)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate-text', methods=['POST'])
def generate_text():
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        sentiment = data.get('sentiment', 'auto')
        length = data.get('length', 'medium')
        use_model = data.get('use_model', False)  
        
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        if sentiment == 'auto':
            sentiment, confidence = get_sentiment(prompt)
        else:
            confidence = 1.0
        
        if use_model:
            generated_text = generate_text_with_model(prompt, sentiment, length)
        else:
            generated_text = generate_text_from_template(prompt, sentiment, length)
        
        word_count = len(generated_text.split())
        
        return jsonify({
            'text': generated_text,
            'sentiment': sentiment,
            'confidence': round(confidence, 3),
            'word_count': word_count,
            'length': length
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'sentiment_model': sentiment_analyzer is not None,
        'text_model': text_generator is not None,
        'status': 'healthy' if sentiment_analyzer else 'degraded'
    })


if __name__ == '__main__':
    print("\n" + "="*50)
    print("AI Sentiment-Based Text Generator")
    print("Backend Server Starting...")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
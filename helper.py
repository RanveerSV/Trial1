import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch
import os
import nltk
from nltk.corpus import wordnet

# Download the WordNet database
#nltk.download('wordnet')
#nltk.download('omw-1.4')

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment" #model trained on twitter data
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


OCEAN_LEXICON = {
    'Openness': [
        'space', 'time', 'atom', 'molecule', 'avagadro', 'quantum', 'theory', 'universe', 'philosophy', 
        'abstract', 'complex', 'mystery', 'future', 'imagine', 'creative', 'art', 'culture', 'diversity', 
        'adventure', 'wonder', 'explore', 'insight', 'knowledge', 'vision', 'weird', 'cool', 'awesome', 'avocado'
    ],
    'Conscientiousness': [
        'plan', 'schedule', 'finish', 'organized', 'detail', 'focus', 'goal', 'habit', 'ready', 'system', 
        'efficiency', 'discipline', 'logic', 'rule', 'prepare', 'deadline', 'work', 'study', 'structure', 
        'clean', 'perfect', 'ambition', 'reliable', 'process', 'careful', 'precise', 'h2o'
    ],
    'Extraversion': [
        'gang', 'yall', 'party', 'everyone', 'excited', 'friends', 'yo', 'hangout', 'energy', 'shout', 
        'together', 'social', 'crowd', 'adventure', 'outgoing', 'laugh', 'lol', 'lmao', 'fun', 'wild', 
        'talkative', 'active', 'presents', 'celebrate', 'vibe', 'broadcast', 'pookies'
    ],
    'Agreeableness': [
        'love', 'please', 'good', 'sorry', 'thanks', 'kind', 'chill', 'trust', 'support', 
        'agree', 'help', 'friendship', 'sweet', 'gentle', 'harmony', 'share', 'care', 'forgive', 
        'sympathy', 'generous', 'warm', 'welcome', 'bless', 'honest', 'appreciate', 'chill'
    ],
    'Neuroticism': [
        'hate', 'sad', 'fuck', 'bitch', 'shit', 'bad', 'worried', 'nervous', 'stress', 'scared', 'maybe', 
        'mood', 'anxious', 'fail', 'angry', 'cry', 'lonely', 'jealous', 'panic', 'threat', 'danger', 
        'pain', 'annoy', 'disgust', 'terrible', 'worst', 'suffering', 'kill', 'murder', 'nigga', 'dickhead'
    ]
}


def parse_chat(file_path):
    if not os.path.exists(file_path):
        print(f"ERROR: File not found at {file_path}")
        return pd.DataFrame()

    # --- UPDATED MULTIFORMAT REGEX ---
    # 1. Handles optional leading '['
    # 2. Matches Date and Time (with optional seconds and AM/PM)
    # 3. Handles optional closing ']' followed by either ' - ' or just a space
    # 4. Captures Sender and Message
    pattern = r'^\[?(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?::\d{2})?\s?[apAPmM\u202f]*)\]?[\s-]*([^:]+):\s(.*)'
    
    parsed_data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # We process line by line to handle multi-line messages correctly
        current_date, current_sender, current_message = None, None, None
        
        for line in f:
            match = re.match(pattern, line)
            if match:
                # If we found a new message, save the previous one first
                if current_sender:
                    parsed_data.append([current_date, current_sender, current_message])
                
                current_date, current_sender, current_message = match.groups()
            else:
                # This handles messages that span multiple lines
                if current_sender:
                    current_message += " " + line.strip()

        # Append the very last message in the file
        if current_sender:
            parsed_data.append([current_date, current_sender, current_message])
    
    df = pd.DataFrame(parsed_data, columns=['Timestamp', 'Sender', 'Message'])
    
    # --- Noise Filtering ---
    noise = ["Messages and calls are end-to-end encrypted", "<Media omitted>", "joined using this group's invite link"]
    df = df[~df['Message'].str.contains('|'.join(noise), case=False, na=False)]
    
    low_signal_reactions = {'ok', 'k', 'yes', 'no', 'lol', 'lmao', 'yo', 'hi', 'hello', 'yep', 'hmm', 'oh'}
    
    def is_meaningful(msg):
        if not isinstance(msg, str): return False
        msg_clean = msg.lower().strip()
        words = msg_clean.split()
        # Keep if it has more than 2 words OR is not in the low_signal list
        return len(words) > 2 and msg_clean not in low_signal_reactions

    df = df[df['Message'].apply(is_meaningful)]
    
    # Group by sender for Personality Analysis
    user_corpus = df.groupby('Sender')['Message'].apply(lambda x: ' '.join(x)).reset_index()
    user_corpus = user_corpus[user_corpus['Message'].str.strip() != ""]
    
    return user_corpus



def expand_query(query):
    words = query.lower().split()
    expanded_words = set(words)
    
    for word in words:
        # Get synonyms from WordNet
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                # Replace underscores with spaces for multi-word synonyms
                expanded_words.add(lemma.name().replace('_', ' '))
                
    return " ".join(list(expanded_words))

def get_sentiment_score(text):
    # to identify irony and sarcasm
    # Truncate to 512 tokens for the model
    inputs = tokenizer(text[:512], return_tensors='pt').to(model.device)
    
    with torch.no_grad():
        output = model(**inputs)
        
    scores = output[0][0].detach().cpu().numpy()
    scores = softmax(scores)
    return scores # [Neg, Neu, Pos]

def get_ocean_vector(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    total = len(words) or 1
    
    vector = []
    for trait in ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']:
        count = sum(1 for w in words if any(k in w for k in OCEAN_LEXICON[trait]))
        vector.append(count / total)
    
    return np.array(vector)

def get_refined_ocean_vector(text):

    base_vector = get_ocean_vector(text) 
    sentiment = get_sentiment_score(text)
    
    # Sarcasm Logic: High negative tone + High Agreeable words = Sarcasm
    if sentiment[0] > 0.6:
        base_vector[3] *= 0.2  
        base_vector[4] *= 1.5  
        
    return base_vector

def find_most_likely(question, corpus):
    # QUERY EXPANSION
    expanded_q = expand_query(question)
    q_lower = question.lower()
    
    # CONTEXT MATCH (TF-IDF)
    vectorizer = TfidfVectorizer(sublinear_tf=True, analyzer='char_wb', ngram_range=(2, 4))
    tfidf_matrix = vectorizer.fit_transform(corpus['Message'])
    q_vec = vectorizer.transform([expanded_q])
    tfidf_scores = cosine_similarity(q_vec, tfidf_matrix).flatten()

    # ARCHTYPES ANALYSIS
    target = np.array([0.2, 0.2, 0.2, 0.2, 0.2]) # Default neutral
    
    if any(w in q_lower for w in ['murder', 'kill', 'bad', 'villain', 'rapist', 'toxic', 'jail']):
        target = np.array([0.1, -0.3, 0.2, -1.0, 1.0]) 
    
    elif any(w in q_lower for w in ['love', 'kiss', 'pookie', 'lover', 'good', 'bestie', 'popular']):
        target = np.array([0.3, 0.2, 1.0, 1.0, -0.5])
        
    elif any(w in q_lower for w in ['smart', 'physicist', 'atom', 'nerd', 'science', 'math', 'expert']):
        target = np.array([1.0, 1.0, -0.2, 0.1, -0.2])
    
    else:
        target = get_ocean_vector(q_lower)

    if np.sum(np.abs(target)) > 0:
        target = target / np.linalg.norm(target)

    # FINAL MATCHING
    user_traits = np.array([get_refined_ocean_vector(m) for m in corpus['Message']])
    pers_scores = cosine_similarity(target.reshape(1,-1), user_traits).flatten()

    # DYNAMIC FUSION
    if np.max(tfidf_scores) > 0.12:
        corpus['Score'] = (tfidf_scores * 0.7) + (pers_scores * 0.3)
    else:
        corpus['Score'] = pers_scores
        
    corpus['Context_Match'] = tfidf_scores
    corpus['Personality_Match'] = pers_scores
    
    return corpus.sort_values(by='Score', ascending=False)


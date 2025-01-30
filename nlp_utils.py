from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
                     'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
                     'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
                     'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
                     'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
                     'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                     'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
                     'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
                     'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                     'further', 'then', 'once'])


def extract_keywords(text):
    """Extract keywords with domain-specific enhancements"""
    keywords = set()
    
    # Add domain-specific terms
    aviation_terms = {
        "fatigue": ["fatigue risk", "crew rest", "safety management"],
        "baggage": ["carry-on", "checked baggage", "luggage"],
        "operations": ["flight operations", "safety protocols", "regulatory compliance"]
    }
    
    for term, synonyms in aviation_terms.items():
        if term in text.lower():
            keywords.update(synonyms)
    
    # Add general keywords
    words = word_tokenize(text.lower())
    keywords.update([word for word in words if word.isalpha() and word not in stopwords])
    
    return list(keywords)

# Optimize the NLP functions for better performance
def generate_summary(text: str) -> str:
    """Optimized summary generation with timeout protection"""
    if not text:
        return ""
    try:
        # Limit input size
        max_length = 10000
        if len(text) > max_length:
            text = text[:max_length]
            
        sentences = sent_tokenize(text)
        if len(sentences) <= 3:
            return text

        # Use numpy for faster computation
        word_freq = Counter(word_tokenize(text.lower()))
        scores = np.zeros(len(sentences))
        
        for i, sentence in enumerate(sentences):
            words = word_tokenize(sentence.lower())
            if words:
                scores[i] = sum(word_freq[word] for word in words if word not in stopwords) / len(words)
        
        # Get indices of top sentences
        top_indices = np.argsort(scores)[-3:]
        top_indices.sort()  # Keep original order
        
        summary = ' '.join(sentences[i] for i in top_indices)
        return summary

    except Exception as e:
        print(f"Error generating summary: {e}")
        return text[:200] + "..."


from sentence_transformers import SentenceTransformer

# Initialize the model ONCE at startup
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_vector_embedding(text):
    """Generate semantic embeddings using Sentence Transformers"""
    if not text:
        return [0.0] * 384  # Default size for 'all-MiniLM-L6-v2'
    try:
        return model.encode(text).tolist()
    except Exception as e:
        print(f"Embedding error: {str(e)}")
        return [0.0] * 384         

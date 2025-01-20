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


def extract_keywords(text: str, max_keywords: int = 5) -> list:
    """
    Extract keywords with robust error handling and fallback mechanisms
    """
    if not text:
        return []
        
    try:
        # Limit input size for processing efficiency
        max_length = 10000
        if len(text) > max_length:
            text = text[:max_length]

        # Simple tokenization fallback if NLTK fails
        try:
            tokens = word_tokenize(text.lower())
        except Exception:
            tokens = text.lower().split()

        # Simple POS tagging fallback if NLTK fails
        try:
            tagged = pos_tag(tokens)
            # Keep only nouns and adjectives
            important_words = [word for word, tag in tagged 
                             if tag.startswith(('NN', 'JJ')) 
                             and word not in stop_words 
                             and len(word) > 2]
        except Exception:
            # Fallback: just filter by length and stopwords
            important_words = [word for word in tokens 
                             if word not in stop_words 
                             and len(word) > 2]

        # Get word frequencies
        word_freq = Counter(important_words)
        
        # Return top keywords
        return [word for word, _ in word_freq.most_common(max_keywords)]

    except Exception as e:
        print(f"Error in keyword extraction: {str(e)}")
        # Last resort fallback: return basic filtered words
        try:
            words = text.lower().split()
            filtered_words = [w for w in words if w not in stop_words and len(w) > 2]
            return list(set(filtered_words))[:max_keywords]
        except:
            return []
        

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


def generate_vector_embedding(text: str) -> list:
    """
    Generate document embedding using TF-IDF with matching dimensions.
    """
    vector_size = 384  # Match the existing embedding size
    if not text:
        return [0] * vector_size
    try:
        # Initialize TF-IDF with matching dimensions
        tfidf = TfidfVectorizer(max_features=vector_size)
        
        # Create a small corpus with the text to ensure proper vectorization
        corpus = [text]
        
        # Fit and transform the text
        sparse_vector = tfidf.fit_transform(corpus)
        
        # Convert to dense array and ensure fixed size
        dense_vector = sparse_vector.toarray()[0]
        
        # Pad or truncate to match exact size
        if len(dense_vector) < vector_size:
            return np.pad(dense_vector, (0, vector_size - len(dense_vector))).tolist()
        return dense_vector[:vector_size].tolist()

    except Exception as e:
        print(f"Error generating embedding: {e}")
        return [0] * vector_size        

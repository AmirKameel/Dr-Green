import gc
from functools import wraps, partial
import os
import re
import threading
import fitz  # PyMuPDF
import openai
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from typing import Dict, Generator, List, Any, Optional
from dotenv import load_dotenv
from database.compliance_reports import ComplianceReports
from database.manual_sections import ManualSections
from nlp_utils import generate_summary, generate_vector_embedding
from database.regulation_sections import RegulationSections
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from database.regulations import Regulations
from database.manuals import Manuals
import json
import math
import ssl
import time
import traceback
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from collections import Counter
# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.config['AEROSYNC_API_KEY'] = os.getenv('AEROSYNC_API_KEY')  
import nltk
nltk.download('punkt_tab')



import os
import ssl
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Create data directory if it doesn't exist
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Set NLTK data path
nltk.data.path.append(nltk_data_dir)

def download_nltk_data():
    """Download required NLTK data with error handling and SSL workaround"""
    try:
        # Handle SSL certificate verification issues
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        # Download required NLTK data
        required_packages = ['punkt', 'averaged_perceptron_tagger', 'stopwords']
        for package in required_packages:
            try:
                nltk.download(package, download_dir=nltk_data_dir, quiet=True)
            except Exception as e:
                print(f"Error downloading {package}: {str(e)}")

    except Exception as e:
        print(f"Error setting up NLTK: {str(e)}")

# Initialize stopwords with fallback
try:
    stop_words = set(stopwords.words('english'))
except Exception:
    # Fallback stopwords if NLTK download fails
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

# Download NLTK data
download_nltk_data()

def extract_keywords(text):
    """Extract aviation-specific keywords"""
    aviation_terms = {
        "FRMS": ["fatigue risk management system", "crew rest", "safety protocols"],
        "EFB": ["electronic flight bag", "flight planning", "PBN"],
        "ADS-B": ["automatic dependent surveillance", "situational awareness", "RVSM"],
        "Multi-crew": ["multi-pilot operations", "cockpit crew", "flight deck team"],
        "A320": ["airbus a320", "fly-by-wire", "advanced avionics"]
    }
    
    keywords = set()
    text_lower = text.lower()
    
    # Match profile-specific terms
    for term, synonyms in aviation_terms.items():
        if term.lower() in text_lower or any(synonym in text_lower for synonym in synonyms):
            keywords.update([term] + synonyms)
    
    # Add general keywords
    words = word_tokenize(text_lower)
    stop_words = set(nltk_stopwords.words('english'))
    keywords.update([word for word in words if word.isalpha() and word not in stop_words])
    
    return list(keywords)


class BatchProcessor:
    def __init__(self, chunk_size=10, max_retries=3, timeout=120):
        self.chunk_size = chunk_size
        self.max_retries = max_retries
        self.timeout = timeout
        self.processed_count = 0
        self.total_sections = 0
        self._lock = threading.Lock()

    def process_sections_in_chunks(self, sections: List[Dict], regulation_id: int) -> Generator[Dict, None, None]:
        """
        Process sections in chunks with progress tracking and memory management
        """
        self.total_sections = len(sections)
        
        for i in range(0, len(sections), self.chunk_size):
            chunk = sections[i:i + self.chunk_size]
            
            # Process chunk with retries
            chunk_results = self._process_chunk(chunk, regulation_id)
            
            # Update progress
            with self._lock:
                self.processed_count += len(chunk)
                progress = (self.processed_count / self.total_sections) * 100
                
            # Clear memory after each chunk
            gc.collect()
            
            yield {
                'results': chunk_results,
                'progress': progress,
                'processed': self.processed_count,
                'total': self.total_sections
            }
            
            # Small delay between chunks
            time.sleep(0.5)

    def _process_chunk(self, chunk: List[Dict], regulation_id: int) -> Dict:
        """
        Process a single chunk with retry logic and memory optimization
        """
        results = {
            'successful': [],
            'failed': []
        }

        for section_data in chunk:
            success = False
            retry_count = 0
            
            while not success and retry_count < self.max_retries:
                try:
                    # Process single section with timeout
                    with ThreadPoolExecutor(max_workers=3) as executor:
                        section_result = self._process_single_section(
                            section_data, 
                            regulation_id,
                            executor
                        )
                        
                    results['successful'].append(section_result)
                    success = True
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count == self.max_retries:
                        results['failed'].append({
                            'section_name': section_data.get('section_name'),
                            'error': str(e)
                        })
                    time.sleep(retry_count)  # Exponential backoff
                    
        return results

    def _process_single_section(self, section_data: Dict, regulation_id: int, executor) -> Dict:
        """
        Process a single section with parallel NLP processing
        """
        full_text = section_data.get('full_text', '')
        if not full_text:
            raise ValueError('No full_text provided')

        # Parallel NLP processing with timeout
        future_summary = executor.submit(generate_summary, full_text)
        future_keywords = executor.submit(extract_keywords, full_text)
        future_embedding = executor.submit(generate_vector_embedding, full_text)
        
        summary = future_summary.result(timeout=self.timeout)
        keywords = future_keywords.result(timeout=self.timeout)
        vector_embedding = future_embedding.result(timeout=self.timeout)
        
        # Create section with optimized text processing
        section = RegulationSections.create_section(
            regulation_id=regulation_id,
            section_name=section_data['section_name'],
            section_number=section_data.get('section_number'),
            full_text=self._optimize_text_storage(full_text),
            summary=summary,
            keywords=keywords,
            vector_embedding=vector_embedding
        )
        
        return {
            'section_id': section.get('id'),
            'section_name': section_data.get('section_name')
        }

    def _optimize_text_storage(self, text: str) -> str:
        """
        Optimize text storage by removing redundant whitespace and normalizing
        """
        # Remove excessive whitespace
        text = ' '.join(text.split())
        # Limit text size if needed
        max_length = 100000  # Adjust based on your database limits
        if len(text) > max_length:
            text = text[:max_length]
        return text

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




# Regulations Endpoints
@app.route('/regulations', methods=['GET'])
def list_regulations():
    regulations = Regulations.list_regulations()
    return jsonify(regulations), 200

@app.route('/regulations/<int:regulation_id>', methods=['GET'])
def get_regulation(regulation_id):
    regulation = Regulations.get_regulation(regulation_id)
    if not regulation:
        return jsonify({"error": "Regulation not found"}), 404
    return jsonify(regulation), 200

@app.route('/regulations', methods=['POST'])
def create_regulation():
    data = request.json
    regulation = Regulations.create_regulation(
        regulation_name=data['regulation_name'],
        metadata=data.get('metadata'),
        category=data.get('category'),
        subcategory=data.get('subcategory')
    )
    return jsonify(regulation), 201

@app.route('/regulations/<int:regulation_id>', methods=['PUT'])
def update_regulation(regulation_id):
    data = request.json
    regulation = Regulations.update_regulation(regulation_id, data)
    return jsonify(regulation), 200

@app.route('/regulations/<int:regulation_id>', methods=['DELETE'])
def delete_regulation(regulation_id):
    Regulations.delete_regulation(regulation_id)
    return jsonify({"message": "Regulation deleted"}), 200

# Regulation Sections Endpoints
@app.route('/regulations/<int:regulation_id>/sections', methods=['GET'])
def list_regulation_sections(regulation_id):
    sections = RegulationSections.list_sections_by_regulation(regulation_id)
    return jsonify(sections), 200

@app.route('/sections/<int:section_id>', methods=['GET'])
def get_section(section_id):
    section = RegulationSections.get_section(section_id)
    if not section:
        return jsonify({"error": "Section not found"}), 404
    return jsonify(section), 200

@app.route('/regulations/<int:regulation_id>/section', methods=['POST'])
def create_section(regulation_id):
    """Create a new regulation section with NLP processing"""
    try:
        data = request.get_json()
        
        # Log incoming data
        print(f"Creating section with data: {json.dumps(data, indent=2)}")
        
        # Process text fields
        full_text = data.get('full_text', '')
        if not full_text:
            return jsonify({'error': 'full_text is required'}), 400
            
        # Generate NLP features
        summary = generate_summary(full_text)
        keywords = extract_keywords(full_text)
        vector_embedding = generate_vector_embedding(full_text)
        
        # Log NLP results
        print(f"Generated features:\nSummary length: {len(summary)}\n"
              f"Keywords: {keywords}\n"
              f"Embedding length: {len(vector_embedding)}")
        
        # Create section in database
        section = RegulationSections.create_section(
            regulation_id=regulation_id,
            section_name=data['section_name'],
            section_number=data.get('section_number'),
            full_text=full_text,
            summary=summary,
            keywords=keywords,
            vector_embedding=vector_embedding
        )
        
        return jsonify(section), 201
        
    except Exception as e:
        print(f"Section creation error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/regulations/<int:regulation_id>/sections', methods=['POST'])
def create_sections(regulation_id):
    """Handle large PDF sections with streaming response and progress tracking"""
    try:
        data = request.get_json()
        
        if not isinstance(data, list):
            data = [data]
            
        if not data:
            return jsonify({'error': 'No sections provided'}), 400

        # Initialize batch processor with optimal chunk size
        total_sections = len(data)
        chunk_size = min(10, math.ceil(total_sections / 20))  # Adjust based on total
        processor = BatchProcessor(chunk_size=chunk_size)

        # Process sections with progress tracking
        all_results = {
            'successful': [],
            'failed': [],
            'total_processed': 0
        }

        # Stream process the sections
        for chunk_result in processor.process_sections_in_chunks(data, regulation_id):
            results = chunk_result['results']
            all_results['successful'].extend(results['successful'])
            all_results['failed'].extend(results['failed'])
            all_results['total_processed'] = chunk_result['processed']
            
            # Log progress
            print(f"Progress: {chunk_result['progress']:.2f}% "
                  f"({chunk_result['processed']}/{chunk_result['total']})")

        # Prepare final response
        response = {
            'message': f'Processed {len(all_results["successful"])} out of {total_sections} sections',
            'successful_sections': len(all_results['successful']),
            'failed_sections': len(all_results['failed']),
            'successful': all_results['successful'],
            'failed': all_results['failed']
        }

        # Determine appropriate status code
        if len(all_results['successful']) == 0:
            return jsonify(response), 400
        elif len(all_results['failed']) > 0:
            return jsonify(response), 207  # Partial Content
        else:
            return jsonify(response), 201

    except Exception as e:
        print(f"Batch section creation error: {str(e)}")
        return jsonify({'error': str(e)}), 500



@app.route('/regulations/<int:regulation_id>/sections/<int:section_id>', methods=['PUT'])
def update_section_with_regulation(regulation_id, section_id):
    data = request.json
    section = RegulationSections.get_section(section_id)
    if not section:
        return jsonify({"error": "Section not found"}), 404
    if section["regulation_id"] != regulation_id:
        return jsonify({"error": "Section does not belong to the specified regulation"}), 400
    updated_section = RegulationSections.update_section(section_id, data)
    return jsonify(updated_section), 200

@app.route('/sections/<int:section_id>', methods=['DELETE'])
def delete_section(section_id):
    RegulationSections.delete_section(section_id)
    return jsonify({"message": "Section deleted"}), 200

# Manuals Endpoints
# Manuals Endpoints
@app.route('/manuals', methods=['GET'])
def list_manuals():
    manuals = Manuals.list_manuals()
    return jsonify(manuals), 200

@app.route('/manuals/<int:manual_id>', methods=['GET'])
def get_manual(manual_id):
    manual = Manuals.get_manual(manual_id)
    if not manual:
        return jsonify({"error": "Manual not found"}), 404
    return jsonify(manual), 200

@app.route('/manuals', methods=['POST'])
def create_manual():
    data = request.json
    manual = Manuals.create_manual(
        manual_name=data['manual_name'],
        metadata=data.get('metadata'),
        category=data.get('category'),
        subcategory=data.get('subcategory')
    )
    return jsonify(manual), 201

@app.route('/manuals/<int:manual_id>', methods=['PUT'])
def update_manual(manual_id):
    data = request.json
    manual = Manuals.update_manual(manual_id, data)
    return jsonify(manual), 200

@app.route('/manuals/<int:manual_id>', methods=['DELETE'])
def delete_manual(manual_id):
    Manuals.delete_manual(manual_id)
    return jsonify({"message": "Manual deleted"}), 200

# Manual Sections Endpoints
@app.route('/manuals/<int:manual_id>/sections', methods=['GET'])
def list_manual_sections(manual_id):
    sections = ManualSections.list_sections_by_manual(manual_id)
    return jsonify(sections), 200

@app.route('/manuals/<int:manual_id>/manual-sections/<int:section_id>', methods=['PUT'])
def update_manual_section_with_manual(manual_id, section_id):
    data = request.json
    section = ManualSections.get_section(section_id)
    if not section:
        return jsonify({"error": "Manual section not found"}), 404
    if section["manual_id"] != manual_id:
        return jsonify({"error": "Manual section does not belong to the specified manual"}), 400
    updated_section = ManualSections.update_section(section_id, data)
    return jsonify(updated_section), 200

@app.route('/manuals/<int:manual_id>/sections', methods=['POST'])
def create_manual_section(manual_id):
    data = request.json
    full_text = data.get('full_text', "")
    summary = generate_summary(full_text)
    keywords = extract_keywords(full_text)
    vector_embedding = generate_vector_embedding(full_text)
    section = ManualSections.create_section(
        manual_id=manual_id,
        section_name=data['section_name'],
        section_number=data.get('section_number'),
        parent_section_id=data.get('parent_section_id'),
        full_text=full_text,
        summary=summary,
        keywords=keywords,
        vector_embedding=vector_embedding
    )
    return jsonify(section), 201

@app.route('/manual-sections/<int:section_id>', methods=['PUT'])
def update_manual_section(section_id):
    data = request.json
    section = ManualSections.update_section(section_id, data)
    return jsonify(section), 200

@app.route('/manual-sections/<int:section_id>', methods=['DELETE'])
def delete_manual_section(section_id):
    ManualSections.delete_section(section_id)
    return jsonify({"message": "Manual section deleted"}), 200


# Compliance Reports Endpoints
@app.route('/compliance-reports', methods=['GET'])
def list_compliance_reports():
    reports = ComplianceReports.list_reports()
    return jsonify(reports), 200

@app.route('/compliance-reports/<int:report_id>', methods=['GET'])
def get_compliance_report(report_id):
    report = ComplianceReports.get_report(report_id)
    if not report:
        return jsonify({"error": "Compliance report not found"}), 404
    return jsonify(report), 200

@app.route('/compliance-reports', methods=['POST'])
def create_compliance_report():
    data = request.json
    report = ComplianceReports.create_report(
        regulation_section_id=data['regulation_section_id'],
        manual_section_id=data['manual_section_id'],
        compliance_score=data['compliance_score'],
        report_text=data['report_text'],
        metadata=data.get('metadata')
    )
    return jsonify(report), 201

@app.route('/compliance-reports/<int:report_id>', methods=['PUT'])
def update_compliance_report(report_id):
    data = request.json
    report = ComplianceReports.update_report(report_id, data)
    return jsonify(report), 200

@app.route('/compliance-reports/<int:report_id>', methods=['DELETE'])
def delete_compliance_report(report_id):
    ComplianceReports.delete_report(report_id)
    return jsonify({"message": "Compliance report deleted"}), 200

# Health Check
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200


#similart
def build_profile_text(profile_data):
    """Convert airline profile to a focused text description"""
    profile_text = []
    
    # Technical profile
    tech = profile_data.get('technical_profile', {})
    if tech.get('FATIGUE_RISK_MANAGEMENT_SYSTEM') == "YES":
        profile_text.append("The airline operates a Fatigue Risk Management System (FRMS) with real-time crew monitoring and onboard rest facilities.")
    if tech.get('USE_OF_ELECTRONIC_FLIGHT_BAG') == "YES":
        profile_text.append("The fleet uses Electronic Flight Bags (EFB) for Performance-Based Navigation (PBN) and advanced flight planning.")
    if tech.get('AUTOMATIC_DEPENDENT_SURVEILLANCE_BROADCAST_ADS_B_OUT_OPERATIONS') == "YES":
        profile_text.append("ADS-B technology is implemented for enhanced surveillance in RVSM airspace.")
    
    # Aircraft details
    if profile_data.get('aircraft_groups', {}).get('Group_4_Total_weight_exceeds_5700_Kg_12500_lbs_Turbo_Jet_powered') == "YES":
        profile_text.append("The airline operates Airbus A320/A321 jets with modern avionics and multi-crew cockpits.")
    
    return ". ".join(profile_text)
    
def calculate_similarity2(section_data, profile_data):
    try:
        # Generate embeddings
        profile_text = build_profile_text(profile_data)
        section_text = section_data.get('full_text', '')
        
        profile_embedding = generate_vector_embedding(profile_text)
        section_embedding = generate_vector_embedding(section_text)
        
        # Cosine similarity
        similarity = np.dot(profile_embedding, section_embedding) / (
            np.linalg.norm(profile_embedding) * np.linalg.norm(section_embedding)
        )
        
        # Keyword matching
        profile_keywords = set(extract_keywords(profile_text))
        section_keywords = set(extract_keywords(section_text))
        matching_keywords = profile_keywords.intersection(section_keywords)
        
        return {
            'total_score': similarity * 0.8 + (len(matching_keywords)/50) * 0.2,  # 80% weight to semantic similarity
            'keyword_score': len(matching_keywords) / 50,  # Normalized to 0-1
            'text_similarity_score': similarity,
            'profile_matches': list(matching_keywords)
        }
    
    except Exception as e:
        print(f"Similarity error: {str(e)}")
        return {'total_score': 0.0, 'keyword_score': 0.0, 'text_similarity_score': 0.0, 'profile_matches': []}


@app.route('/analyze-section-similarity', methods=['POST'])
def analyze_section_similarity():
    """
    Analyze similarity between a given section and an airline profile.
    Generates embeddings dynamically for both the section and profile.
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()
        
        # Check for required fields
        if 'section' not in data or 'airline_profile' not in data:
            return jsonify({"error": "Missing required fields (section and airline_profile)"}), 400

        section_data = data['section']
        profile_data = data['airline_profile']

        # Log incoming request
        print(f"Analyzing section similarity with profile:\n{json.dumps(profile_data, indent=2)}")

        # Generate embeddings for the section
        section_text = section_data.get('full_text', '')
        if not section_text:
            return jsonify({"error": "Section text is required"}), 400
        section_embedding = generate_vector_embedding(section_text)

        # Generate embeddings for the airline profile
        profile_text = build_profile_text(profile_data)
        profile_embedding = generate_vector_embedding(profile_text)

        # Log embeddings for debugging
        print(f"Section Embedding: {section_embedding}")
        print(f"Profile Embedding: {profile_embedding}")

        # Calculate similarity
        similarity_scores = calculate_similarity2(
            {
                'section_name': section_data.get('section_name', 'Unnamed Section'),
                'full_text': section_text,
                'vector_embedding': section_embedding,
                'keywords': extract_keywords(section_text)
            },
            profile_data
        )

        # Prepare response
        response = {
            'section_name': section_data.get('section_name', 'Unnamed Section'),
            'similarity_scores': similarity_scores,
            'relevance_explanation': (
                f"Section '{section_data.get('section_name', 'Unnamed Section')}' has a "
                f"similarity score of {similarity_scores['total_score']:.2f}"
            )
        }

        return jsonify(response), 200

    except Exception as e:
        print(f"Error analyzing section similarity: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def calculate_weighted_similarity(section_data, profile_data):
    """Calculate similarity with weighted profile fields"""
    weights = {
        'technical_profile': 0.6,
        'composition_of_crew': 0.2,
        'aircraft_groups': 0.2
    }
    
    total_score = 0.0
    
    # Calculate similarity for each profile part
    for part, weight in weights.items():
        if part in profile_data:
            part_text = build_profile_text({part: profile_data[part]})
            part_embedding = generate_vector_embedding(part_text)
            section_embedding = section_data.get('vector_embedding', [])
            
            if section_embedding:
                similarity = np.dot(part_embedding, section_embedding) / (
                    np.linalg.norm(part_embedding) * np.linalg.norm(section_embedding)
                )
                total_score += similarity * weight
    
    return total_score





def calculate_similarity(section_data, profile_data):
    """Calculate similarity between section and profile with semantic embeddings"""
    try:
        # Build profile text
        profile_text = build_profile_text(profile_data)
        print(f"Generated profile text: {profile_text[:100]}...")
        
        # Generate embeddings with validation
        profile_embedding = generate_vector_embedding(profile_text)
        section_embedding = section_data.get('vector_embedding', [])
        
        # Ensure section_embedding is a list of floats
        if isinstance(section_embedding, str):
            section_embedding = json.loads(section_embedding)
        
        # Validate embeddings
        if not profile_embedding or not section_embedding:
            print("Warning: Missing embeddings")
            return {
                'total_score': 0.0,
                'keyword_score': 0.0,
                'text_similarity_score': 0.0,
                'profile_matches': []
            }

        # Convert to numpy arrays and ensure float32
        profile_vector = np.array(profile_embedding, dtype=np.float32)
        section_vector = np.array(section_embedding, dtype=np.float32)
        
        print(f"Profile vector shape: {profile_vector.shape}")
        print(f"Section vector shape: {section_vector.shape}")
        
        # Calculate cosine similarity
        dot_product = np.dot(profile_vector, section_vector)
        profile_norm = np.linalg.norm(profile_vector)
        section_norm = np.linalg.norm(section_vector)
        
        if profile_norm == 0 or section_norm == 0:
            similarity = 0.0
        else:
            similarity = dot_product / (profile_norm * section_norm)
            
        # Get keywords for additional scoring
        profile_keywords = set(extract_keywords(profile_text))
        section_keywords = set(section_data.get('keywords', []))
        matching_keywords = profile_keywords.intersection(section_keywords)
        
        keyword_score = len(matching_keywords) / max(len(section_keywords), 1)
        
        # Calculate weighted score
        total_score = (similarity * 0.7) + (keyword_score * 0.3)
        
        print(f"Similarity scores for section {section_data.get('section_name')}:")
        print(f"Text similarity: {similarity:.4f}")
        print(f"Keyword score: {keyword_score:.4f}")
        print(f"Total score: {total_score:.4f}")
        
        return {
            'total_score': float(total_score),
            'keyword_score': float(keyword_score),
            'text_similarity_score': float(similarity),
            'profile_matches': list(matching_keywords)
        }

    except Exception as e:
        print(f"Error calculating similarity: {str(e)}")
        traceback.print_exc()
        return {
            'total_score': 0.0,
            'keyword_score': 0.0,
            'text_similarity_score': 0.0,
            'profile_matches': []
        }

@app.route('/analyze-regulation-similarity/<int:regulation_id>', methods=['POST'])
def analyze_regulation_similarity(regulation_id):
    """Analyze similarity between regulation sections and airline profile"""
    try:
        profile_data = request.get_json()
        if not profile_data:
            return jsonify({'error': 'No profile data provided'}), 400
            
        # Log incoming request
        print(f"Analyzing regulation {regulation_id} with profile:\n"
              f"{json.dumps(profile_data, indent=2)}")
        
        # Get all sections for the regulation
        sections = RegulationSections.list_sections_by_regulation(regulation_id)
        if not sections:
            print(f"No sections found for regulation {regulation_id}")
            return jsonify({
                'regulation_id': regulation_id,
                'analysis_results': [],
                'message': 'No sections found'
            }), 200
            
        print(f"Found {len(sections)} sections")
        
        # Calculate similarity for each section
        results = []
        for section in sections:
            similarity_scores = calculate_similarity(section, profile_data)
            
            # Only include sections with meaningful similarity
            if similarity_scores['total_score'] > 0.1:  # Minimum threshold
                result = {
                    'section_id': section.get('id'),
                    'section_name': section.get('section_name'),
                    'section_number': section.get('section_number'),
                    'summary': section.get('summary', ''),
                    'similarity_scores': similarity_scores,
                    'relevance_explanation': (
                        f"Section '{section.get('section_name')}' has a "
                        f"similarity score of {similarity_scores['total_score']:.2f}"
                    )
                }
                results.append(result)
        
        # Sort by total score
        results.sort(key=lambda x: x['similarity_scores']['total_score'], reverse=True)
        
        return jsonify({
            'regulation_id': regulation_id,
            'analysis_results': results,
            'total_matches': len(results)
        }), 200
        
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def require_api_key(f):
    """
    Decorator to require API key for routes
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check if API key is in headers
        api_key = request.headers.get('AeroSync-API-Key')
        
        if not api_key:
            return jsonify({"error": "No API key provided"}), 401
            
        if api_key != app.config['AEROSYNC_API_KEY']:
            return jsonify({"error": "Invalid API key"}), 403
            
        return f(*args, **kwargs)
    return decorated_function

from typing import Dict, Any
import json
from datetime import datetime

class FlexibleAuditProcessor:
    def __init__(self, compliance_reports_db, regulation_sections_db, manual_sections_db):
        self.compliance_reports_db = compliance_reports_db
        self.regulation_sections_db = regulation_sections_db
        self.manual_sections_db = manual_sections_db

    def process_and_store_audit(self, 
                              audit_text: str,
                              regulation_section_id: Optional[int] = None,
                              manual_section_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Process audit results and store with flexible section IDs
        """
        try:
            # Parse audit results
            parsed_result = self._parse_audit_result(audit_text)
            
            # Try to identify relevant sections if IDs not provided
            if not regulation_section_id or not manual_section_id:
                identified_sections = self._identify_relevant_sections(audit_text)
                regulation_section_id = regulation_section_id or identified_sections.get('regulation_id')
                manual_section_id = manual_section_id or identified_sections.get('manual_id')

            # Create metadata
            metadata = {
                "timestamp": datetime.utcnow().isoformat(),
                "findings": parsed_result.get("findings", []),
                "recommendations": parsed_result.get("recommendations", []),
                "status": parsed_result.get("status", "completed"),
                "regulation_section_id": regulation_section_id,
                "manual_section_id": manual_section_id,
                "confidence_score": parsed_result.get("confidence_score", 0.0)
            }

            # Store report
            report = self.compliance_reports_db.create_report(
                regulation_section_id=regulation_section_id,
                manual_section_id=manual_section_id,
                compliance_score=parsed_result.get("compliance_score", 0.0),
                report_text=audit_text,
                metadata=metadata
            )

            return {
                "status": "success",
                "report_id": report.get("id"),
                "identified_sections": {
                    "regulation_section_id": regulation_section_id,
                    "manual_section_id": manual_section_id
                },
                "compliance_score": parsed_result.get("compliance_score", 0.0),
                "findings": parsed_result.get("findings", []),
                "recommendations": parsed_result.get("recommendations", [])
            }

        except Exception as e:
            print(f"Error in audit processing: {e}")
            return {
                "status": "error",
                "message": str(e),
                "partial_results": parsed_result if 'parsed_result' in locals() else None
            }

    def _parse_audit_result(self, audit_text: str) -> Dict[str, Any]:
        """
        Enhanced parsing of audit results with confidence scoring
        """
        structured_data = {
            "compliance_score": 0.0,
            "confidence_score": 0.0,
            "findings": [],
            "recommendations": [],
            "status": "completed",
            "identified_references": []
        }

        try:
            # Extract compliance score
            score_match = re.search(r"compliance score:?\s*(\d+\.?\d*)", audit_text.lower())
            if score_match:
                structured_data["compliance_score"] = float(score_match.group(1))

            # Extract findings
            findings_section = re.split(r"findings:|non-compliance:", audit_text.lower(), maxsplit=1)
            if len(findings_section) > 1:
                findings = [f.strip() for f in findings_section[1].split("\n") if f.strip()]
                structured_data["findings"] = findings

            # Extract recommendations
            recommendations_section = re.split(r"recommendations:", audit_text.lower(), maxsplit=1)
            if len(recommendations_section) > 1:
                recommendations = [r.strip() for r in recommendations_section[1].split("\n") if r.strip()]
                structured_data["recommendations"] = recommendations

            # Extract references to regulations and manuals
            structured_data["identified_references"] = self._extract_references(audit_text)
            
            # Calculate confidence score based on completeness
            structured_data["confidence_score"] = self._calculate_confidence_score(structured_data)

            return structured_data

        except Exception as e:
            print(f"Error parsing audit result: {e}")
            return structured_data

    def _identify_relevant_sections(self, audit_text: str) -> Dict[str, Optional[int]]:
        """
        Identify relevant section IDs from audit text
        """
        identified_sections = {
            'regulation_id': None,
            'manual_id': None
        }

        try:
            # Extract potential section references
            references = self._extract_references(audit_text)
            
            # Try to match references to database records
            for ref in references:
                # Check regulations
                reg_section = self.regulation_sections_db.find_section_by_name(
                    section_name=ref
                )
                if reg_section:
                    identified_sections['regulation_id'] = reg_section.get('id')
                    continue

                # Check manuals
                manual_section = self.manual_sections_db.find_section_by_name(
                    section_name=ref
                )
                if manual_section:
                    identified_sections['manual_id'] = manual_section.get('id')

        except Exception as e:
            print(f"Error identifying sections: {e}")

        return identified_sections

    def _extract_references(self, text: str) -> list:
        """
        Extract regulation and manual references from text
        """
        references = []
        
        # Common patterns for references
        patterns = [
            r'ORG \d+(\.\d+)*',  # e.g., ORG 1.1.1
            r'ISM \d+(\.\d+)*',  # e.g., ISM 2.1
            r'Section \d+(\.\d+)*',  # e.g., Section 3.1.2
            r'[A-Z]{2,4} \d+(\.\d+)*'  # e.g., IOSA 1.2
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            references.extend(matches)

        return list(set(references))

    def _calculate_confidence_score(self, parsed_data: Dict) -> float:
        """
        Calculate confidence score based on completeness of parsed data
        """
        score = 0.0
        total_weights = 0.0

        # Define weights for different components
        weights = {
            'compliance_score': 0.3,
            'findings': 0.3,
            'recommendations': 0.2,
            'identified_references': 0.2
        }

        # Score each component
        if parsed_data.get('compliance_score', 0) > 0:
            score += weights['compliance_score']
        total_weights += weights['compliance_score']

        if parsed_data.get('findings'):
            score += weights['findings']
        total_weights += weights['findings']

        if parsed_data.get('recommendations'):
            score += weights['recommendations']
        total_weights += weights['recommendations']

        if parsed_data.get('identified_references'):
            score += weights['identified_references']
        total_weights += weights['identified_references']

        return (score / total_weights) * 100 if total_weights > 0 else 0.0





import os
from typing import Dict, List
import fitz  # PyMuPDF
import re
from collections import defaultdict
import time
from tenacity import retry, stop_after_attempt, wait_exponential

class PDFProcessor:
    def __init__(self, chunk_size=5, max_workers=2, max_retries=3):
        self.chunk_size = chunk_size  # Reduced chunk size for better stability
        self.max_workers = max_workers  # Reduced workers to prevent overload
        self.max_retries = max_retries
        self.processed_count = 0
        self.total_sections = 0
        self._lock = threading.Lock()
        self.section_counter = defaultdict(int)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry_error_callback=lambda retry_state: {"error": "Max retries reached"}
    )
    def _extract_page_text(self, doc, page_num: int) -> str:
        """
        Extract text from a page with retry logic
        """
        try:
            page = doc.load_page(page_num)
            text = page.get_text("text")
            if not text:
                text = page.get_text("blocks")
            return text if text else ""
        except Exception as e:
            print(f"Error extracting text from page {page_num}: {str(e)}")
            raise

    def extract_sections(self, pdf_path: str, expand_pages: int = 7) -> List[Dict]:
        """
        Extract sections from PDF with improved error handling
        """
        sections = []
        seen_sections = defaultdict(int)

        try:
            doc = fitz.open(pdf_path)
            toc = doc.get_toc()

            # Process TOC entries
            for level, title, page in toc:
                try:
                    section_text = ""
                    success = False
                    retry_count = 0

                    while not success and retry_count < self.max_retries:
                        try:
                            for i in range(page - 1, min(page - 1 + expand_pages + 1, len(doc))):
                                page_text = self._extract_page_text(doc, i)
                                section_text += page_text

                            success = True
                        except Exception as e:
                            retry_count += 1
                            if retry_count < self.max_retries:
                                time.sleep(2 ** retry_count)  # Exponential backoff
                            else:
                                print(f"Failed to extract section {title} after {self.max_retries} attempts")
                                section_text = f"Error extracting text: {str(e)}"

                    # Handle duplicate section names
                    base_title = title
                    seen_sections[base_title] += 1
                    if seen_sections[base_title] > 1:
                        title = f"{base_title} (Version {seen_sections[base_title]})"

                    sections.append({
                        "section_name": title,
                        "section_number": str(level),
                        "full_text": section_text.strip(),
                        "original_title": base_title,
                        "extraction_status": "success" if success else "partial_failure"
                    })

                except Exception as e:
                    print(f"Error processing section {title}: {str(e)}")
                    sections.append({
                        "section_name": title,
                        "section_number": str(level),
                        "full_text": f"Error processing section: {str(e)}",
                        "original_title": title,
                        "extraction_status": "failure"
                    })

            # Process regex patterns with similar retry logic
            pattern = r'\b(ORG \d+(\.\d+){1,5})\b'
            for page_num in range(len(doc)):
                try:
                    page_text = self._extract_page_text(doc, page_num)
                    headers = re.findall(pattern, page_text)

                    for header in headers:
                        base_header_title = header[0]
                        seen_sections[base_header_title] += 1
                        header_title = base_header_title
                        
                        if seen_sections[base_header_title] > 1:
                            header_title = f"{base_header_title} (Version {seen_sections[base_header_title]})"

                        section_text = ""
                        success = False
                        retry_count = 0

                        while not success and retry_count < self.max_retries:
                            try:
                                for i in range(page_num, min(page_num + expand_pages + 1, len(doc))):
                                    page_text = self._extract_page_text(doc, i)
                                    section_text += page_text
                                success = True
                            except Exception as e:
                                retry_count += 1
                                if retry_count < self.max_retries:
                                    time.sleep(2 ** retry_count)
                                else:
                                    section_text = f"Error extracting text: {str(e)}"

                        sections.append({
                            "section_name": header_title,
                            "section_number": str(base_header_title.count('.') + 1),
                            "full_text": section_text.strip(),
                            "original_title": base_header_title,
                            "extraction_status": "success" if success else "partial_failure"
                        })

                except Exception as e:
                    print(f"Error processing page {page_num}: {str(e)}")

            return sections

        except Exception as e:
            print(f"Error extracting sections: {str(e)}")
            return []
        finally:
            if 'doc' in locals():
                doc.close()

    def _process_batch(self, batch: List[Dict], regulation_id: int) -> Dict:
        """
        Process a batch of sections with improved error handling
        """
        batch_results = {
            'successful': [],
            'failed': [],
            'partial': []
        }

        for section_data in batch:
            try:
                # Skip processing if extraction failed
                if section_data.get('extraction_status') == 'failure':
                    batch_results['failed'].append({
                        'section_name': section_data.get('section_name'),
                        'error': 'Text extraction failed'
                    })
                    continue

                # Process text and generate features
                full_text = section_data.get('full_text', '')
                if full_text.startswith('Error'):
                    batch_results['failed'].append({
                        'section_name': section_data.get('section_name'),
                        'error': full_text
                    })
                    continue

                # Add delay between processing to prevent overload
                time.sleep(0.1)

                summary = generate_summary(full_text)
                keywords = extract_keywords(full_text)
                vector_embedding = generate_vector_embedding(full_text)

                with self._lock:
                    self.section_counter[section_data['original_title']] += 1
                    current_count = self.section_counter[section_data['original_title']]

                final_section_name = section_data['section_name']

                section = RegulationSections.create_section(
                    regulation_id=regulation_id,
                    section_name=final_section_name,
                    section_number=section_data.get('section_number'),
                    full_text=full_text,
                    summary=summary,
                    keywords=keywords,
                    vector_embedding=vector_embedding,
                    original_title=section_data['original_title']
                )

                status = 'successful'
                if section_data.get('extraction_status') == 'partial_failure':
                    status = 'partial'
                    batch_results['partial'].append({
                        'section_name': final_section_name,
                        'id': section.get('id'),
                        'message': 'Partial content extracted'
                    })
                else:
                    batch_results['successful'].append({
                        'section_name': final_section_name,
                        'id': section.get('id'),
                        'action': 'created'
                    })

                with self._lock:
                    self.processed_count += 1
                    progress = (self.processed_count / self.total_sections) * 100
                    print(f"Progress: {progress:.2f}% ({self.processed_count}/{self.total_sections}) - Status: {status}")

            except Exception as e:
                print(f"Error processing section {section_data.get('section_name')}: {str(e)}")
                batch_results['failed'].append({
                    'section_name': section_data.get('section_name'),
                    'error': str(e)
                })

        return batch_results

    def process_sections(self, sections: List[Dict], regulation_id: int) -> Dict:
        """
        Process sections in batches with parallel processing
        """
        self.total_sections = len(sections)
        self.processed_count = 0
        results = {
            'successful': [],
            'failed': [],
            'total_processed': 0
        }

        # Process sections in batches
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for i in range(0, len(sections), self.chunk_size):
                batch = sections[i:i + self.chunk_size]
                future = executor.submit(self._process_batch, batch, regulation_id)
                futures.append(future)

            # Collect results
            for future in futures:
                batch_result = future.result()
                results['successful'].extend(batch_result['successful'])
                results['failed'].extend(batch_result['failed'])
                results['total_processed'] += len(batch_result['successful'])

        return results

def perform_audit(iosa_checklist: str, input_text: str) -> str:
    """
    Perform an audit using GPT to evaluate compliance with ISARPs
    
    Args:
        iosa_checklist (str): The IOSA checklist standards to evaluate against
        input_text (str): The text to be evaluated
        airline_profile (str): The airline profile information
        
    Returns:
        str: Audit results including assessment, recommendations, and compliance scores
    """
    
    

    # Create the OpenAI client using the API key from secrets.toml
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    # OpenAI API request
    # OpenAI API request
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                'role': 'system',
                'content': (
                    """
    You are an expert aviation auditor with over 20 years of experience in both business and commercial aviation. Your expertise encompasses a deep understanding of aviation regulations, safety protocols, and the regulatory differences that exist internationally. Your role involves conducting thorough audits of airlines to ensure compliance with ISARPs and other regulatory standards.

    As an auditor, your focus is not just on the technical aspects of compliance but also on the operational features unique to different airlines. You understand the nuances of airline profiles, including fleet characteristics, operational environments, and the specific regulatory obligations that apply to various types of operations (e.g., passenger, cargo, charter).

    Your assessments are meticulous, using precise terminology and a formal tone to convey findings and recommendations. You evaluate documentation against established standards, identifying strengths and weaknesses, and providing actionable insights for improvement.

    In your evaluations, consider factors such as:
    - Operational complexity and safety management systems.
    - Training and competency frameworks for flight and cabin crew.
    - Maintenance and engineering practices aligned with regulatory requirements.
    - Customer service protocols and their impact on operational safety and compliance.
    """
                )
            },
            {
                'role': 'user',
                'content': (
                    f"""
    OBJECTIVES:
    You are provided with a document and an input text. Please perform the following actions:
    The provided text is to be evaluated on a compliance scale against the requirements of the regulatory text or international standard, ranging from 0 to 10. A score of 0 indicates the text is entirely non-compliant or irrelevant to the set requirements, while a score of 10 denotes full compliance with the specified criteria.

    Analyze the text's relevance and adherence to the given standards, assigning an appropriate score within this range based on your assessment. Justify the score thoroughly, detailing how the text meets or fails to meet the established requirements.

    If the compliance score exceeds 3, provide supplementary text that draws from industry best practices and regulatory standards, crafted in a clear, readable style suitable for crew members. This should emphasize human factors principles for better understanding.

    If the provided text is entirely irrelevant, leverage your expertise and industry benchmarks to create a detailed exposition on processes, procedures, or organizational structures within the aviation industry, revising the text to ensure compliance with applicable legal requirements.

    ISARPs: 
    {iosa_checklist}
    INPUT_TEXT: 
    {input_text}

    Your output must include the following sections:
    ASSESSMENT: A detailed evaluation of the documentation's alignment with the ISARPs, using technical language and aviation terminology.
    RECOMMENDATIONS: Specific, actionable suggestions for improving compliance with ISARP standards, in a formal and professional tone.
    OVERALL_COMPLIANCE_SCORE: A numerical rating (0 to 10) reflecting the documentation's overall compliance with the ISARPs.
    OVERALL_COMPLIANCE_TAG: A scoring tag indicating the overall compliance level with ISARPs.
    """
                )
            }
        ],
        temperature=0.3,        
        max_tokens=4000
    )
    
    return response.choices[0].message.content
# Modified audit endpoint
@app.route('/audit', methods=['POST'])
@require_api_key
def conduct_audit():
    """
    Flexible API endpoint for real-time aviation compliance auditing
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
            
        data = request.json
        
        # Check for required audit input
        if 'iosa_checklist' not in data or 'input_text' not in data:
            return jsonify({
                "error": "Missing required audit input (iosa_checklist and input_text required)"
            }), 400

        # Get optional section IDs
        regulation_section_id = data.get('regulation_section_id')
        manual_section_id = data.get('manual_section_id')

        # Perform the audit
        audit_result = perform_audit(
            data['iosa_checklist'],
            data['input_text']
        )

        # Process and store results
        audit_processor = FlexibleAuditProcessor(
            compliance_reports_db=ComplianceReports,
            regulation_sections_db=RegulationSections,
            manual_sections_db=ManualSections
        )
        
        processing_result = audit_processor.process_and_store_audit(
            audit_text=audit_result,
            regulation_section_id=regulation_section_id,
            manual_section_id=manual_section_id
        )

        # Return comprehensive response
        return jsonify({
            "message": "Audit completed successfully",
            "audit_result": audit_result,
            "processing_result": processing_result,
            "identified_sections": processing_result.get("identified_sections", {}),
            "confidence_score": processing_result.get("confidence_score", 0.0)
        }), 200

    except Exception as e:
        print(f"Audit error: {str(e)}")
        return jsonify({
            "error": "Error performing audit",
            "details": str(e)
        }), 500
    
@app.route('/auditv1', methods=['POST'])
@require_api_key
def audit():
    """
    API endpoint to perform aviation compliance audit.
    Returns the audit result as plain text.
    """
    try:
        # Validate request data
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.json
        required_fields = ['iosa_checklist', 'input_text']

        # Check if all required fields are present
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400

        # Perform the audit
        audit_result = perform_audit(
            data['iosa_checklist'],
            data['input_text']
        )

        # Return the audit result as plain text
        return audit_result, 200, {'Content-Type': 'text/plain'}

    except Exception as e:
        return jsonify({
            "error": f"Error performing audit: {str(e)}"
        }), 500




def extract_toc_and_sections(pdf_path: str, expand_pages: int = 7) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract table of contents and sections from a PDF file
    
    Args:
        pdf_path (str): Path to the PDF file
        expand_pages (int, optional): Number of pages to expand for section extraction. Defaults to 7.
    
    Returns:
        Dict containing extracted sections
    """
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()  # Extract the Table of Contents (TOC)
    sections = {}

    # Create a dictionary to map TOC entries to text in the PDF
    for toc_entry in toc:
        level, title, page = toc_entry
        try:
            # Extract text from the starting page and the following pages
            section_text = ""
            for i in range(page - 1, min(page - 1 + expand_pages + 1, len(doc))):
                page_text = doc.load_page(i).get_text("text")
                if not page_text:
                    page_text = doc.load_page(i).get_text("blocks")  # Try blocks if text is empty
                section_text += page_text if page_text else "Text not available for this section\n"
            
            # Check if the title already exists in sections, if so append to the list
            if title in sections:
                sections[title].append({
                    "level": level,
                    "page": page,
                    "text": section_text.strip()
                })
            else:
                sections[title] = [{
                    "level": level,
                    "page": page,
                    "text": section_text.strip()
                }]
        except Exception as e:
            if title in sections:
                sections[title].append({
                    "level": level,
                    "page": page,
                    "text": f"Error extracting text: {str(e)}"
                })
            else:
                sections[title] = [{
                    "level": level,
                    "page": page,
                    "text": f"Error extracting text: {str(e)}"
                }]

    # Function to detect section headers like "ORG 1.1.1", "ORG 2.3.4", etc.
    def find_section_headers(page_text):
        pattern = r'\b(ORG \d+(\.\d+){1,5})\b'  # Matches patterns like ORG 1.1, ORG 2.1.1, etc.
        headers = re.findall(pattern, page_text)
        return [header[0] for header in headers]

    # Scan each page for section headers not in the TOC
    for page_num in range(len(doc)):
        page_text = doc.load_page(page_num).get_text("text")
        headers = find_section_headers(page_text)

        for header in headers:
            # If header is not already in sections, add it
            section_text = ""
            for i in range(page_num, min(page_num + expand_pages + 1, len(doc))):
                page_text = doc.load_page(i).get_text("text")
                if not page_text:
                    page_text = doc.load_page(i).get_text("blocks")  # Try blocks if text is empty
                section_text += page_text if page_text else "Text not available for this section\n"
            
            # Append this occurrence of the header to the list in sections
            if header in sections:
                sections[header].append({
                    "level": header.count('.') + 1,  # Determine level by the number of dots
                    "page": page_num + 1,
                    "text": section_text.strip()
                })
            else:
                sections[header] = [{
                    "level": header.count('.') + 1,
                    "page": page_num + 1,
                    "text": section_text.strip()
                }]

    return sections

def extract_section_with_gpt(section_name: str, chunk_text: str) -> str:
    """
    Extract a specific section from text using GPT
    
    Args:
        section_name (str): Name of the section to extract
        chunk_text (str): Text chunk to extract from
    
    Returns:
        str: Extracted section text
    """
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # OpenAI API request
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {
                'role': 'system',
                'content': (
                    """
    Context:
    You are tasked with extracting sections from a document. Your focus is on finding specific sections based on their header names and extracting only the relevant portion. Ignore any unrelated text that appears before or after the specified section. If you select a parent section, extract all its child sections. If it's a child without subchildren, extract only that section.
                    """
                )
            },
            {
                'role': 'user',
                'content': (
                    f"""
    OBJECTIVE:
    You are provided with the full text of a document. Your task is to extract the section titled "{section_name}". The section starts with this title and ends at the conclusion of the relevant content. Please extract and return only the content of the section titled "{section_name}".

    Here is the document text:
    {chunk_text}

    Extract and return only the content of the section titled "{section_name}". Do not include unrelated text.
                    """
                )
            }
        ],
        max_tokens=4000  # Adjust token limit based on document size
    )

    # Return the extracted section
    return response.choices[0].message.content

@app.route('/upload', methods=['POST'])
@require_api_key
def upload_pdf():
    """
    API endpoint to upload PDF and extract sections
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Extract sections
            sections = extract_toc_and_sections(filepath)
            
            # Optional: Clean up the uploaded file
            os.remove(filepath)
            
            return jsonify({
                "message": "PDF processed successfully",
                "sections": sections
            }), 200
        
        except Exception as e:
            # Optional: Clean up the uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
            
            return jsonify({"error": str(e)}), 500


# Add this new endpoint to your Flask app
@app.route('/regulations/<int:regulation_id>/upload-document', methods=['POST'])
def upload_and_process_document(regulation_id):
    """
    Upload PDF document and process its sections into the regulation
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Create upload directory if it doesn't exist
        upload_dir = 'uploads'
        os.makedirs(upload_dir, exist_ok=True)

        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(upload_dir, filename)
        file.save(filepath)

        # Initialize processor and extract sections
        processor = PDFProcessor(chunk_size=10, max_workers=4)
        sections = processor.extract_sections(filepath)

        # Process and store sections
        results = processor.process_sections(sections, regulation_id)

        # Clean up uploaded file
        os.remove(filepath)

        return jsonify({
            'message': 'Document processed successfully',
            'total_sections': len(sections),
            'successful_sections': len(results['successful']),
            'failed_sections': len(results['failed']),
            'failures': results['failed']
        }), 200

    except Exception as e:
        # Clean up file if it exists
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500



@app.route('/extract_section', methods=['POST'])
@require_api_key
def extract_section():
    """
    API endpoint to extract a specific section
    """
    data = request.json
    
    if not data or 'section_name' not in data or 'text' not in data:
        return jsonify({"error": "Invalid request. Requires section_name and text"}), 400
    
    try:
        extracted_text = extract_section_with_gpt(
            data['section_name'], 
            data['text']
        )
        
        return jsonify({
            "section_name": data['section_name'],
            "extracted_text": extracted_text
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500





# Error Handlers
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File too large"}), 413

@app.errorhandler(400)
def bad_request(error):
    return jsonify({"error": "Bad request"}), 400

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    if not app.config['AEROSYNC_API_KEY']:
        raise ValueError("API_KEY environment variable must be set")
    app.run(debug=True, host='0.0.0.0', port=8000)

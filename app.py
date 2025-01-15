from concurrent.futures import ThreadPoolExecutor
import json
import math
import time
import traceback
from typing import Dict, List
from flask import Flask, request, jsonify
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from collections import Counter
from database.regulations import Regulations
from database.regulation_sections import RegulationSections
from database.manuals import Manuals
from database.manual_sections import ManualSections
from database.compliance_reports import ComplianceReports

app = Flask(__name__)

# Download required NLTK data at startup
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Initialize global variables
stop_words = set(stopwords.words('english'))
tfidf = TfidfVectorizer(max_features=100)  # Limit features for memory efficiency


def process_section_chunk(sections: List[Dict], regulation_id: int, chunk_size: int = 5) -> Dict:
    """
    Process a chunk of sections with built-in retry logic.
    """
    results = {
        'successful': [],
        'failed': []
    }
    
    for i in range(0, len(sections), chunk_size):
        chunk = sections[i:i + chunk_size]
        
        for section_data in chunk:
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    # Extract required fields
                    full_text = section_data.get('full_text', '')
                    if not full_text:
                        results['failed'].append({
                            'section_name': section_data.get('section_name'),
                            'error': 'No full_text provided'
                        })
                        break
                        
                    # Generate NLP features with timeout protection
                    with ThreadPoolExecutor() as executor:
                        future_summary = executor.submit(generate_summary, full_text)
                        future_keywords = executor.submit(extract_keywords, full_text)
                        future_embedding = executor.submit(generate_vector_embedding, full_text)
                        
                        # Set timeout for NLP processing
                        summary = future_summary.result(timeout=30)
                        keywords = future_keywords.result(timeout=30)
                        vector_embedding = future_embedding.result(timeout=30)
                    
                    # Create section in database
                    section = RegulationSections.create_section(
                        regulation_id=regulation_id,
                        section_name=section_data['section_name'],
                        section_number=section_data.get('section_number'),
                        full_text=full_text,
                        summary=summary,
                        keywords=keywords,
                        vector_embedding=vector_embedding
                    )
                    
                    results['successful'].append({
                        'section_id': section.get('id'),
                        'section_name': section_data.get('section_name')
                    })
                    break
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        results['failed'].append({
                            'section_name': section_data.get('section_name'),
                            'error': str(e)
                        })
                    time.sleep(1)  # Wait before retry
                    
        # Add a small delay between chunks to prevent overwhelming the server
        time.sleep(0.5)
    
    return results

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
                scores[i] = sum(word_freq[word] for word in words if word not in stop_words) / len(words)
        
        # Get indices of top sentences
        top_indices = np.argsort(scores)[-3:]
        top_indices.sort()  # Keep original order
        
        summary = ' '.join(sentences[i] for i in top_indices)
        return summary

    except Exception as e:
        print(f"Error generating summary: {e}")
        return text[:200] + "..."

def extract_keywords(text: str) -> list:
    """Optimized keyword extraction"""
    if not text:
        return []
    try:
        # Limit input size
        max_length = 10000
        if len(text) > max_length:
            text = text[:max_length]
            
        # More efficient tokenization
        tokens = word_tokenize(text.lower())
        tagged = pos_tag(tokens)
        
        # Use set operations for faster filtering
        important_words = {word for word, tag in tagged 
                         if tag.startswith(('NN', 'JJ')) 
                         and word not in stop_words 
                         and len(word) > 2}
        
        return list(Counter(important_words).most_common(5))

    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return []

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
    """Create multiple regulation sections with chunked processing and progress tracking"""
    try:
        data = request.get_json()
        
        # Check if data is a list or single object
        if not isinstance(data, list):
            data = [data]
            
        if not data:
            return jsonify({'error': 'No sections provided'}), 400
            
        # Calculate optimal chunk size based on data length
        total_sections = len(data)
        chunk_size = min(5, math.ceil(total_sections / 10))  # Adjust chunk size based on total
        
        # Process sections in chunks
        results = process_section_chunk(data, regulation_id, chunk_size)
        
        # Prepare response
        response = {
            'message': f'Processed {len(results["successful"])} out of {total_sections} sections',
            'successful_sections': len(results['successful']),
            'failed_sections': len(results['failed']),
            'successful': results['successful'],
            'failed': results['failed']
        }
        
        # Determine appropriate status code
        if len(results['successful']) == 0:
            return jsonify(response), 400
        elif len(results['failed']) > 0:
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





def calculate_similarity(section_data, profile_data):
    """Calculate similarity between section and profile with fixed dimensions"""
    try:
        scores = {
            'total_score': 0.0,
            'keyword_score': 0.0,
            'text_similarity_score': 0.0,
            'profile_matches': []
        }

        # Build profile text
        profile_parts = []
        if profile_data.get('aircraft_type'):
            profile_parts.append(f"Aircraft type: {profile_data['aircraft_type']}")
        if profile_data.get('type_of_operation'):
            profile_parts.append(f"Operation type: {profile_data['type_of_operation']}")

        ops_specs = profile_data.get('ops_specs', {})
        if ops_specs:
            enabled_specs = [f"{k}: {v}" for k, v in ops_specs.items()]
            profile_parts.extend(enabled_specs)

        profile_text = ' '.join(profile_parts)

        # Calculate keyword similarity
        profile_keywords = set(extract_keywords(profile_text))
        section_keywords = set(section_data.get('keywords', []))
        
        if section_keywords and profile_keywords:
            matching_keywords = section_keywords.intersection(profile_keywords)
            scores['keyword_score'] = len(matching_keywords) / max(len(section_keywords), 1)
            scores['profile_matches'] = list(matching_keywords)

        # Calculate text similarity using fixed-dimension vectors
        section_embedding = section_data.get('vector_embedding', [])
        if isinstance(section_embedding, str):
            try:
                section_embedding = eval(section_embedding)
            except:
                section_embedding = []

        if section_embedding:
            profile_embedding = generate_vector_embedding(profile_text)
            
            # Ensure both embeddings have the same dimension
            if len(section_embedding) != len(profile_embedding):
                print(f"Warning: Embedding dimension mismatch. Section: {len(section_embedding)}, Profile: {len(profile_embedding)}")
                # Pad the shorter embedding if necessary
                max_dim = max(len(section_embedding), len(profile_embedding))
                section_embedding = np.pad(section_embedding, (0, max_dim - len(section_embedding)))
                profile_embedding = np.pad(profile_embedding, (0, max_dim - len(profile_embedding)))
            
            # Calculate cosine similarity
            dot_product = np.dot(section_embedding, profile_embedding)
            norm_product = (np.linalg.norm(section_embedding) * 
                          np.linalg.norm(profile_embedding))
            
            if norm_product != 0:
                scores['text_similarity_score'] = float(dot_product / norm_product)

        # Calculate total score
        scores['total_score'] = (scores['keyword_score'] * 0.5 + 
                               scores['text_similarity_score'] * 0.5)

        return scores

    except Exception as e:
        print(f"Similarity calculation error: {str(e)}")
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
            try:
                similarity_scores = calculate_similarity(section, profile_data)
                
                result = {
                    'section_id': section.get('id'),
                    'section_name': section.get('section_name'),
                    'section_number': section.get('section_number'),
                    'summary': section.get('summary', ''),
                    'similarity_scores': similarity_scores,
                    'relevance_explanation': (
                        f"Section '{section.get('section_name')}' has a "
                        f"total similarity score of {similarity_scores['total_score']:.2f}."
                    )
                }
                results.append(result)
                
            except Exception as e:
                print(f"Error processing section {section.get('id')}: {str(e)}")
                continue
        
        # Sort by similarity score
        results.sort(key=lambda x: x['similarity_scores']['total_score'], reverse=True)
        
        return jsonify({
            'regulation_id': regulation_id,
            'analysis_results': results
        }), 200
        
    except Exception as e:
        print(f"Analysis error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

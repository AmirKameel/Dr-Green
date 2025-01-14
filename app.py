import json
import traceback
from django import db
from flask import Flask, request, jsonify
import torch
from database.regulations import Regulations
from database.regulation_sections import RegulationSections
from database.manuals import Manuals
from database.manual_sections import ManualSections
from database.compliance_reports import ComplianceReports
from transformers import pipeline  # For summarization
from sentence_transformers import SentenceTransformer  # For embeddings
from keybert import KeyBERT  # For keyword extraction

app = Flask(__name__)

# Initialize lightweight models
summarizer = pipeline("summarization", model="t5-small")  # Smaller summarization model
# Initialize the SentenceTransformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Smaller embedding model
kw_model = KeyBERT(model=model)  # Use the same model for keyword extraction



# Helper functions
def generate_summary(text: str) -> str:
    """
    Generate a summary of the input text using a lightweight model.
    """
    if not text:
        return ""
    try:
        summary = summarizer(text, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
        return summary
    except Exception as e:
        print(f"Error generating summary: {e}")
        return ""

def extract_keywords(text: str) -> list:
    """
    Extract keywords from the input text using a lightweight model.
    """
    if not text:
        return []
    try:
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=5)
        return [kw[0] for kw in keywords]  # Return only the keywords, not their scores
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return []

def generate_vector_embedding(text: str) -> list:
    """
    Generate a vector embedding for the input text using a lightweight model.
    """
    if not text:
        return []
    try:
        embedding = model.encode(text)
        return embedding.tolist()  # Convert numpy array to list of floats
    except Exception as e:
        print(f"Error generating vector embedding: {e}")
        return []

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

@app.route('/regulations/<int:regulation_id>/sections', methods=['POST'])
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

from flask import Flask, request, jsonify
from sentence_transformers import util
import numpy as np



def calculate_similarity(section_data, profile_data):
    """Calculate similarity between section and profile with improved logic"""
    try:
        # Initialize scores
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

        # Add OPS SPECS
        ops_specs = profile_data.get('ops_specs', {})
        if ops_specs:
            enabled_specs = [f"{k}: {v}" for k, v in ops_specs.items()]
            profile_parts.extend(enabled_specs)

        profile_text = ' '.join(profile_parts)
        print(f"Profile text: {profile_text}")  # Debugging

        # Extract keywords from profile text
        profile_keywords = set(extract_keywords(profile_text))
        section_keywords = set(section_data.get('keywords', []))
        print(f"Profile keywords: {profile_keywords}")  # Debugging
        print(f"Section keywords: {section_keywords}")  # Debugging

        # Calculate keyword similarity
        if section_keywords and profile_keywords:
            matching_keywords = section_keywords.intersection(profile_keywords)
            scores['keyword_score'] = len(matching_keywords) / max(len(section_keywords), 1)
            scores['profile_matches'] = list(matching_keywords)
            print(f"Matching keywords: {matching_keywords}")  # Debugging
            print(f"Keyword score: {scores['keyword_score']}")  # Debugging

        # Fetch vector embedding from the database
        section_embedding = section_data.get('vector_embedding')
        if section_embedding:
            print(f"Section embedding (before conversion): {section_embedding}")  # Debugging
            # Convert embedding to list of floats if it's a string
            if isinstance(section_embedding, str):
                try:
                    section_embedding = json.loads(section_embedding)  # Convert string to list
                except json.JSONDecodeError:
                    print("Error: Invalid JSON format for section embedding.")
                    section_embedding = []
            print(f"Section embedding (after conversion): {section_embedding}")  # Debugging

            # Generate embedding for the profile text
            profile_embedding = generate_vector_embedding(profile_text)
            if profile_embedding:
                print(f"Profile embedding: {profile_embedding}")  # Debugging
                # Convert embeddings to numerical tensors
                section_embedding_tensor = torch.tensor(section_embedding, dtype=torch.float32).unsqueeze(0)
                profile_embedding_tensor = torch.tensor(profile_embedding, dtype=torch.float32).unsqueeze(0)

                # Calculate cosine similarity
                similarity = util.cos_sim(section_embedding_tensor, profile_embedding_tensor)
                scores['text_similarity_score'] = float(similarity[0][0])
                print(f"Text similarity score: {scores['text_similarity_score']}")  # Debugging

        # Calculate total score with dynamic weighting
        if scores['keyword_score'] > 0 and scores['text_similarity_score'] > 0:
            # If both scores are positive, use weighted sum
            scores['total_score'] = (
                scores['keyword_score'] * 0.5 +
                scores['text_similarity_score'] * 0.5
            )
        else:
            # If one score is zero, prioritize the other
            scores['total_score'] = max(scores['keyword_score'], scores['text_similarity_score'])

        print(f"Total score: {scores['total_score']}")  # Debugging
        return scores

    except Exception as e:
        print(f"Similarity calculation error: {str(e)}\n{traceback.format_exc()}")
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
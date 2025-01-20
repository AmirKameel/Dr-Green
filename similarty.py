from functools import wraps
import os
import re
import threading
import fitz  # PyMuPDF
import openai
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from database.compliance_reports import ComplianceReports
from database.manual_sections import ManualSections
from nlp_utils import extract_keywords, generate_summary, generate_vector_embedding
from database.regulation_sections import RegulationSections
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.config['AEROSYNC_API_KEY'] = os.getenv('AEROSYNC_API_KEY')  

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





class PDFProcessor:
    def __init__(self, chunk_size=10, max_workers=4):
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.processed_count = 0
        self.total_sections = 0
        self._lock = threading.Lock()

    def extract_sections(self, pdf_path: str, expand_pages: int = 7) -> List[Dict]:
        """
        Extract sections from PDF and format them for database storage
        """
        try:
            doc = fitz.open(pdf_path)
            toc = doc.get_toc()
            sections = []

            # Process TOC entries
            for level, title, page in toc:
                try:
                    section_text = ""
                    for i in range(page - 1, min(page - 1 + expand_pages + 1, len(doc))):
                        page_text = doc.load_page(i).get_text("text")
                        if not page_text:
                            page_text = doc.load_page(i).get_text("blocks")
                        section_text += page_text if page_text else ""

                    sections.append({
                        "section_name": title,
                        "section_number": str(level),
                        "full_text": section_text.strip()
                    })

                except Exception as e:
                    print(f"Error processing section {title}: {str(e)}")

            # Find additional sections using regex patterns
            pattern = r'\b(ORG \d+(\.\d+){1,5})\b'
            for page_num in range(len(doc)):
                page_text = doc.load_page(page_num).get_text("text")
                headers = re.findall(pattern, page_text)

                for header in headers:
                    section_text = ""
                    header_title = header[0]
                    
                    # Extract text for found section
                    for i in range(page_num, min(page_num + expand_pages + 1, len(doc))):
                        page_text = doc.load_page(i).get_text("text")
                        if not page_text:
                            page_text = doc.load_page(i).get_text("blocks")
                        section_text += page_text if page_text else ""

                    sections.append({
                        "section_name": header_title,
                        "section_number": str(header_title.count('.') + 1),
                        "full_text": section_text.strip()
                    })

            return sections

        except Exception as e:
            print(f"Error extracting sections: {str(e)}")
            return []

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

    def _process_batch(self, batch: List[Dict], regulation_id: int) -> Dict:
        """
        Process a batch of sections with duplicate handling
        """
        batch_results = {
            'successful': [],
            'failed': []
        }

        for section_data in batch:
            try:
                # Process text and generate features
                full_text = section_data.get('full_text', '')
                summary = generate_summary(full_text)
                keywords = extract_keywords(full_text)
                vector_embedding = generate_vector_embedding(full_text)

                # Try to find existing section
                existing_section = RegulationSections.find_section_by_name(
                    regulation_id=regulation_id,
                    section_name=section_data['section_name']
                )

                if existing_section:
                    # Update existing section
                    section = RegulationSections.update_section(
                        section_id=existing_section['id'],
                        data={
                            'section_number': section_data.get('section_number'),
                            'full_text': full_text,
                            'summary': summary,
                            'keywords': keywords,
                            'vector_embedding': vector_embedding
                        }
                    )
                    action = 'updated'
                else:
                    # Create new section
                    section = RegulationSections.create_section(
                        regulation_id=regulation_id,
                        section_name=section_data['section_name'],
                        section_number=section_data.get('section_number'),
                        full_text=full_text,
                        summary=summary,
                        keywords=keywords,
                        vector_embedding=vector_embedding
                    )
                    action = 'created'

                batch_results['successful'].append({
                    'section_name': section_data['section_name'],
                    'id': section.get('id'),
                    'action': action
                })

                # Update progress
                with self._lock:
                    self.processed_count += 1
                    progress = (self.processed_count / self.total_sections) * 100
                    print(f"Progress: {progress:.2f}% ({self.processed_count}/{self.total_sections})")

            except Exception as e:
                print(f"Error processing section {section_data.get('section_name')}: {str(e)}")
                batch_results['failed'].append({
                    'section_name': section_data.get('section_name'),
                    'error': str(e)
                })

        return batch_results


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




@app.route('/health', methods=['GET'])
@require_api_key
def health_check():
    """
    Health check endpoint
    """
    return jsonify({"status": "healthy"}), 200

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

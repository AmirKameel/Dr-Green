# Using Python requests library
import requests
import json

# API endpoint
BASE_URL = "http://127.0.0.1:5000/"

# Example 1: Analyzing regulation similarity
def analyze_regulation_similarity(regulation_id, airline_profile):
    url = f"{BASE_URL}/analyze-regulation-similarity/{regulation_id}"
    response = requests.post(url, json=airline_profile)
    return response.json()

# Sample airline profile data
airline_profile = {
    "state_of_registration": "United States",
    "state_of_operations": "United States",
    "aircraft_type": "Group 4",  # Turbo Jet > 5700 Kg
    "crew_composition": "Multi crew pilots",
    "type_of_operation": "Scheduled flights",
    "ops_specs": {
        "FATIGUE RISK MANAGEMENT SYSTEM": "YES",
        "CARRY-ON BAGGAGE PROGRAM": "YES",
        "EXTENDED OVER‑WATER OPERATIONS": "YES",
        "OPERATIONS DURING GROUND ICING CONDITIONS": "YES",
        "ELECTRONIC RECORDKEEPING SYSTEM": "YES",
        "TRANSPORTATION OF DANGEROUS GOODS BY AIR": "NO",
        "USE OF ELECTRONIC FLIGHT BAG": "YES",
        "AUTOMATIC DEPENDENT SURVEILLANCE": "YES",
        "ETOPS WITH TWO ENGINE AIRPLANES": "YES",
        "OPERATIONS IN REDUCED VERTICAL SEPARATION MINIMUM": "YES",
        "CATEGORY II INSTRUMENT APPROACH": "YES",
        "CATEGORY III INSTRUMENT APPROACH": "YES"
    },
    "commercial_profile": {
        "subscription": "Compliance + Recommendations",
        "applicable_regulations": ["Local Regulations", "EU", "IOSA"],
        "iosa_sections": [
            "Organization and Management System",
            "Flight Operations",
            "Operational Control and Flight Dispatch",
            "Aircraft Engineering and Maintenance"
        ]
    }
}

# Example regulation ID
regulation_id = 123

# Make the API call
try:
    results = analyze_regulation_similarity(regulation_id, airline_profile)
    print(json.dumps(results, indent=2))
except requests.exceptions.RequestException as e:
    print(f"Error making API call: {e}")

"""
Example Response:
{
    "regulation_id": 123,
    "analysis_results": [
        {
            "section_id": 456,
            "section_name": "Aircraft Operations Requirements",
            "section_number": "121.345",
            "similarity_scores": {
                "total_score": 0.85,
                "keyword_score": 0.80,
                "text_similarity_score": 0.88,
                "profile_matches": [
                    "ETOPS",
                    "RVSM",
                    "Category III",
                    "electronic flight bag"
                ]
            },
            "summary": "Requirements for extended range operations with two-engine airplanes (ETOPS) and operations in RVSM airspace.",
            "relevance_explanation": "Directly addresses ETOPS WITH TWO ENGINE AIRPLANES | Directly addresses OPERATIONS IN REDUCED VERTICAL SEPARATION MINIMUM | Content highly relevant to airline's operational context"
        },
        {
            "section_id": 457,
            "section_name": "Crew Requirements",
            "section_number": "121.346",
            "similarity_scores": {
                "total_score": 0.72,
                "keyword_score": 0.65,
                "text_similarity_score": 0.77,
                "profile_matches": [
                    "fatigue management",
                    "multi crew",
                    "flight operations"
                ]
            },
            "summary": "Requirements for crew composition and fatigue risk management systems.",
            "relevance_explanation": "Directly addresses FATIGUE RISK MANAGEMENT SYSTEM | Matches key operational aspects: multi crew, flight operations"
        }
        // ... more sections ...
    ]
}
"""

# Example 2: Creating a new regulation section
def create_regulation_section(regulation_id, section_data):
    url = f"{BASE_URL}/regulations/{regulation_id}/sections"
    response = requests.post(url, json=section_data)
    return response.json()

# Sample section data
new_section = {
    "section_name": "ETOPS Operations Requirements",
    "section_number": "121.375",
    "parent_section_id": None,
    "full_text": """
    Requirements for Extended-range Twin-engine Operational Performance Standards (ETOPS):
    (a) No certificate holder may conduct ETOPS with an airplane that has two engines unless:
        (1) The airplane is type design-approved for ETOPS;
        (2) The certificate holder's maintenance program includes all ETOPS requirements;
        (3) The certificate holder's operations manual contains ETOPS policies and procedures;
    (b) The certificate holder must comply with the requirements of this section and...
    """,
}

# Create new section
try:
    created_section = create_regulation_section(regulation_id, new_section)
    print(json.dumps(created_section, indent=2))
except requests.exceptions.RequestException as e:
    print(f"Error creating section: {e}")

"""
Example Response for Created Section:
{
    "id": 458,
    "regulation_id": 123,
    "section_name": "ETOPS Operations Requirements",
    "section_number": "121.375",
    "parent_section_id": null,
    "full_text": "Requirements for Extended-range Twin-engine Operational Performance Standards...",
    "summary": "Requirements for conducting ETOPS operations including aircraft certification, maintenance program, and operational procedures.",
    "keywords": ["ETOPS", "twin-engine", "maintenance", "operations manual", "certification"],
    "vector_embedding": [...],  # Vector of floating point numbers
    "created_at": "2025-01-13T10:30:00Z",
    "updated_at": "2025-01-13T10:30:00Z"
}
"""
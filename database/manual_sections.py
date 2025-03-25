from utils.supabase_client import supabase
from typing import Dict

class ManualSections:
    @staticmethod
    def create_section(
        manual_id: int,
        level: int,
        page_number: int,
        order_index: int,        
        section_name: str,
        section_number: str = None,
        parent_section_id: int = None,
        full_text: str = "",
        summary: str = None,
        keywords: list = None,
        vector_embedding: list = None
    ):
        """
        Insert a new section into the manual_sections table.
        """
        data = {
            "manual_id": manual_id,
            "section_name": section_name,
            "section_number": section_number,
            "parent_section_id": parent_section_id,
            "full_text": full_text,
            "summary": summary,
            "keywords": keywords,
            "vector_embedding": vector_embedding,
            "level": level,
            "order_index": order_index,
            "page_number": page_number
        }
        response = supabase.table("manual_sections").insert(data).execute()
        return response.data[0] if response.data else None

    @staticmethod
    def get_section(section_id: int):
        """
        Fetch a section by its ID.
        """
        response = supabase.table("manual_sections").select("*").eq("section_id", section_id).execute()
        return response.data[0] if response.data else None

    @staticmethod
    def update_section(section_id: int, data: Dict) -> Dict:
        """
    Update an existing section
    """
        try:
            response = supabase.table('manual_sections')\
                .update(data)\
                .eq('section_id', section_id)\
                .execute()
        
            if response.data and len(response.data) > 0:
                return response.data[0]
            raise Exception("Failed to update section")

        except Exception as e:
            print(f"Error updating section: {str(e)}")
            raise


    @staticmethod
    def delete_section(section_id: int):
        """
        Delete a section.
        """
        response = supabase.table("manual_sections").delete().eq("section_id", section_id).execute()
        return response.data[0] if response.data else None

    @staticmethod
    def list_sections_by_manual(manual_id: int):
        """
        List all sections for a specific manual.
        """
        response = supabase.table("manual_sections").select("*").eq("manual_id", manual_id).execute()
        return response.data

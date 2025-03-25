from typing import Dict
from utils.supabase_client import supabase

class RegulationSections:
    @staticmethod
    def create_section(
        regulation_id: int,
        section_name: str,
        level: int,
        page_number: int,
        order_index: int,
        section_number: str = None,
        parent_section_id: int = None,
        full_text: str = "",
        summary: str = None,
        keywords: list = None,
        category: str = None,
        subcategory: str = None,
        vector_embedding: list = None,
        original_title: str = None
    ):
        """
        Insert a new section into the regulation_sections table.
        
        Args:
            regulation_id (int): ID of the regulation.
            section_name (str): Name of the section.
            section_number (str, optional): Number of the section. Defaults to None.
            parent_section_id (int, optional): ID of the parent section. Defaults to None.
            full_text (str, optional): Full text of the section. Defaults to "".
            summary (str, optional): Summary of the section. Defaults to None.
            keywords (list, optional): List of keywords. Defaults to None.
            category (str, optional): Category of the section. Defaults to None.
            subcategory (str, optional): Subcategory of the section. Defaults to None.
            vector_embedding (list, optional): Vector embedding of the section. Defaults to None.
        
        Returns:
            dict: The created section record.
        """
        data = {
            "regulation_id": regulation_id,
            "section_name": section_name,
            "section_number": section_number,
            "parent_section_id": parent_section_id,
            "full_text": full_text,
            "summary": summary,
            "keywords": keywords,
            "category": category,
            "subcategory": subcategory,
            "vector_embedding": vector_embedding,
            "original_title" : original_title,
            "level": level,
            "order_index": order_index,
            "page_number": page_number
        }
        response = supabase.table("regulation_sections").insert(data).execute()
        return response.data[0] if response.data else None

    @staticmethod
    def get_section(section_id: int):
        """
        Fetch a section by its ID.
        
        Args:
            section_id (int): ID of the section.
        
        Returns:
            dict: The section record.
        """
        response = supabase.table("regulation_sections").select("*").eq("section_id", section_id).execute()
        return response.data[0] if response.data else None

    @staticmethod
    def delete_section(section_id: int):
        """
        Delete a section.
        
        Args:
            section_id (int): ID of the section.
        
        Returns:
            dict: The deleted section record.
        """
        response = supabase.table("regulation_sections").delete().eq("section_id", section_id).execute()
        return response.data[0] if response.data else None

    @staticmethod
    def list_sections_by_regulation(regulation_id: int):
        """
        Fetch all sections for a specific regulation.
        """
        try:
            response = supabase.table("regulation_sections").select("*").eq("regulation_id", regulation_id).execute()
            return response.data if response.data else []  # Return an empty list if response.data is None
        except Exception as e:
            print(f"Error fetching sections: {e}")
            return []  # Return an empty list in case of an error
        
    @staticmethod
    def find_section_by_name(regulation_id: int, section_name: str) -> Dict:
        """
        Find a section by regulation_id and section_name
        """
        try:
            # Assuming you have a supabase client instance
            response = supabase.table('regulation_sections')\
                .select('*')\
                .eq('regulation_id', regulation_id)\
                .eq('section_name', section_name)\
                .execute()
            
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None

        except Exception as e:
            print(f"Error finding section: {str(e)}")
            return None

    @staticmethod
    def update_section(section_id: int, data: Dict) -> Dict:
        """
    Update an existing section
    """
        try:
            response = supabase.table('regulation_sections')\
                .update(data)\
                .eq('section_id', section_id)\
                .execute()
        
            if response.data and len(response.data) > 0:
                return response.data[0]
            raise Exception("Failed to update section")

        except Exception as e:
            print(f"Error updating section: {str(e)}")
            raise

   

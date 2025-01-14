from utils.supabase_client import supabase

class RegulationSections:
    @staticmethod
    def create_section(
        regulation_id: int,
        section_name: str,
        section_number: str = None,
        parent_section_id: int = None,
        full_text: str = "",
        summary: str = None,
        keywords: list = None,
        category: str = None,
        subcategory: str = None,
        vector_embedding: list = None
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
            "vector_embedding": vector_embedding
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
    def update_section(section_id: int, updates: dict):
        """
        Update a section.
        
        Args:
            section_id (int): ID of the section.
            updates (dict): Fields to update.
        
        Returns:
            dict: The updated section record.
        """
        response = supabase.table("regulation_sections").update(updates).eq("section_id", section_id).execute()
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
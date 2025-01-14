from utils.supabase_client import supabase

class Regulations:
    @staticmethod
    def create_regulation(regulation_name: str, metadata: dict = None, category: str = None, subcategory: str = None):
        """
        Insert a new regulation into the regulations table.
        
        Args:
            regulation_name (str): Name of the regulation.
            metadata (dict, optional): Additional metadata. Defaults to None.
            category (str, optional): Category of the regulation. Defaults to None.
            subcategory (str, optional): Subcategory of the regulation. Defaults to None.
        
        Returns:
            dict: The created regulation record.
        """
        data = {
            "regulation_name": regulation_name,
            "metadata": metadata,
            "category": category,
            "subcategory": subcategory
        }
        response = supabase.table("regulations").insert(data).execute()
        return response.data[0] if response.data else None

    @staticmethod
    def get_regulation(regulation_id: int):
        """
        Fetch a regulation by its ID.
        
        Args:
            regulation_id (int): ID of the regulation.
        
        Returns:
            dict: The regulation record.
        """
        response = supabase.table("regulations").select("*").eq("regulation_id", regulation_id).execute()
        return response.data[0] if response.data else None

    @staticmethod
    def update_regulation(regulation_id: int, updates: dict):
        """
        Update a regulation.
        
        Args:
            regulation_id (int): ID of the regulation.
            updates (dict): Fields to update.
        
        Returns:
            dict: The updated regulation record.
        """
        response = supabase.table("regulations").update(updates).eq("regulation_id", regulation_id).execute()
        return response.data[0] if response.data else None

    @staticmethod
    def delete_regulation(regulation_id: int):
        """
        Delete a regulation.
        
        Args:
            regulation_id (int): ID of the regulation.
        
        Returns:
            dict: The deleted regulation record.
        """
        response = supabase.table("regulations").delete().eq("regulation_id", regulation_id).execute()
        return response.data[0] if response.data else None

    @staticmethod
    def list_regulations():
        """
        List all regulations.
        
        Returns:
            list: A list of all regulation records.
        """
        response = supabase.table("regulations").select("*").execute()
        return response.data
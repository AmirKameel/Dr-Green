from utils.supabase_client import supabase

class Manuals:
    @staticmethod
    def create_manual(manual_name: str, metadata: dict = None, category: str = None, subcategory: str = None):
        """
        Insert a new manual into the manuals table.
        """
        data = {
            "manual_name": manual_name,
            "metadata": metadata,
            "category": category,
            "subcategory": subcategory
        }
        response = supabase.table("manuals").insert(data).execute()
        return response.data[0] if response.data else None

    @staticmethod
    def get_manual(manual_id: int):
        """
        Fetch a manual by its ID.
        """
        response = supabase.table("manuals").select("*").eq("manual_id", manual_id).execute()
        return response.data[0] if response.data else None

    @staticmethod
    def update_manual(manual_id: int, updates: dict):
        """
        Update a manual.
        """
        response = supabase.table("manuals").update(updates).eq("manual_id", manual_id).execute()
        return response.data[0] if response.data else None

    @staticmethod
    def delete_manual(manual_id: int):
        """
        Delete a manual.
        """
        response = supabase.table("manuals").delete().eq("manual_id", manual_id).execute()
        return response.data[0] if response.data else None

    @staticmethod
    def list_manuals():
        """
        List all manuals.
        """
        response = supabase.table("manuals").select("*").execute()
        return response.data
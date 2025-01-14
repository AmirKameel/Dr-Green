from utils.supabase_client import supabase

class ComplianceReports:
    @staticmethod
    def create_report(regulation_section_id: int, manual_section_id: int, compliance_score: float, report_text: str, metadata: dict = None):
        """Insert a new compliance report."""
        data = {
            "regulation_section_id": regulation_section_id,
            "manual_section_id": manual_section_id,
            "compliance_score": compliance_score,
            "report_text": report_text,
            "metadata": metadata
        }
        response = supabase.table("compliance_reports").insert(data).execute()
        return response.data[0] if response.data else None

    @staticmethod
    def get_report(report_id: int):
        """Fetch a report by its ID."""
        response = supabase.table("compliance_reports").select("*").eq("report_id", report_id).execute()
        return response.data[0] if response.data else None

    @staticmethod
    def update_report(report_id: int, updates: dict):
        """Update a report."""
        response = supabase.table("compliance_reports").update(updates).eq("report_id", report_id).execute()
        return response.data[0] if response.data else None

    @staticmethod
    def delete_report(report_id: int):
        """Delete a report."""
        response = supabase.table("compliance_reports").delete().eq("report_id", report_id).execute()
        return response.data[0] if response.data else None

    @staticmethod
    def list_reports():
        """List all compliance reports."""
        response = supabase.table("compliance_reports").select("*").execute()
        return response.data
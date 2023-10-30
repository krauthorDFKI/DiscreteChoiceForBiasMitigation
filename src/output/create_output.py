from src.output.create_tables import create_tables
from src.output.create_plots import create_plots

def create_output():
    """Create output."""
    print("Creating output...")
    create_plots()
    create_tables()
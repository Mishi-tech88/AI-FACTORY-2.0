# src/report_generator.py
import io
from fpdf import FPDF
from datetime import datetime
import os

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'AI Factory Decision Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 0, 'C')

def generate_report(report_data):
    """
    Generate a PDF report from the given data.
    report_data: dict containing all information.
    Returns: bytes of the PDF.
    """
    pdf = PDFReport()
    pdf.add_page()

    # Title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Factory Incident / Decision Report', 0, 1, 'C')
    pdf.ln(5)

    # Decision
    decision = report_data['decision']
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, 'AI Recommendation:', 0, 1)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, f"Action: {decision['action']}")
    pdf.multi_cell(0, 6, f"Reasoning: {decision['reasoning']}")
    pdf.cell(0, 6, f"Confidence: {decision['confidence']:.2f}", 0, 1)
    pdf.ln(3)

    # Explanations
    explanations = report_data.get('explanations', {})
    if explanations:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'Explanations:', 0, 1)
        pdf.set_font('Arial', '', 10)
        summary = explanations.get('summary', 'No explanation available.')
        pdf.multi_cell(0, 5, summary)

    # Human Review
    human_record = report_data.get('human_record', None)
    if human_record:
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'Human Supervisor Decision:', 0, 1)
        pdf.set_font('Arial', '', 11)
        pdf.cell(0, 6, f"Status: {human_record['status'].upper()}", 0, 1)
        if human_record['modified_action']:
            pdf.cell(0, 6, f"Modified Action: {human_record['modified_action']}", 0, 1)
        if human_record['feedback']:
            pdf.cell(0, 6, f"Feedback: {human_record['feedback']}", 0, 1)
        pdf.cell(0, 6, f"Timestamp: {human_record['timestamp']}", 0, 1)

    # Simulation results
    simulation = report_data.get('simulation', {})
    if simulation:
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'Digital Twin Simulation:', 0, 1)
        pdf.set_font('Arial', '', 10)
        df_sim = simulation.get('dataframe', None)
        if df_sim is not None:
            # Convert DataFrame to text
            sim_text = df_sim.to_string(index=False)
            pdf.multi_cell(0, 5, sim_text)
        best = simulation.get('best_scenario', 'N/A')
        pdf.cell(0, 6, f"Best Scenario: {best}", 0, 1)

    # Add image if provided (optional)
    image_path = report_data.get('image_path')
    if image_path and os.path.exists(image_path):
        try:
            pdf.ln(5)
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, 'Analyzed Product Image:', 0, 1)
            pdf.image(image_path, x=10, y=pdf.get_y(), w=80)
        except Exception as e:
            pdf.cell(0, 6, f"Image could not be embedded: {e}", 0, 1)

    # Signature line
    pdf.ln(20)
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 6, 'Approved by: ___________________________', 0, 1)
    pdf.cell(0, 6, 'Date: ___________________________', 0, 1)

    # Output to bytes
    pdf_output = pdf.output(dest='S').encode('latin1')
    return pdf_output
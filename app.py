# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
from PIL import Image
import io
import tempfile

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import the agent system
from src.agent_system import run_agent_system
from src.report_generator import generate_report  # we'll create this next

st.set_page_config(page_title="AI Factory Command Center", layout="wide")
st.title("🏭 AI Factory Intelligence Command Center")

# Sidebar for inputs
st.sidebar.header("Input Data")

# Upload image
uploaded_image = st.sidebar.file_uploader("Upload Product Image", type=["png", "jpg", "jpeg"])
image_path = None
if uploaded_image:
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
        tmp.write(uploaded_image.getvalue())
        image_path = tmp.name
    st.sidebar.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

# Upload sensor CSV (optional)
uploaded_sensor = st.sidebar.file_uploader("Upload Sensor Data (CSV)", type=["csv"])
sensor_df = None
if uploaded_sensor:
    sensor_df = pd.read_csv(uploaded_sensor)
    st.sidebar.write("Sensor data preview:")
    st.sidebar.dataframe(sensor_df.head(5))
else:
    # Generate dummy sensor sequence
    sensor_seq = [{'vibration': 0.5 + 0.1*i + np.random.normal(0,0.1),
                   'pressure': 100 + 2*np.sin(i/3) + np.random.normal(0,1),
                   'rpm': 3000 - 10*i + np.random.normal(0,20)} for i in range(24)]
    sensor_df = pd.DataFrame(sensor_seq)
    st.sidebar.info("Using dummy sensor data. Upload your own CSV for real data.")

# Run button
if st.sidebar.button("Run AI Analysis"):
    if not image_path:
        st.error("Please upload a product image.")
    else:
        with st.spinner("Running all agents... This may take a moment."):
            # Convert sensor_df to list of dicts
            sensor_sequence = sensor_df.to_dict('records')
            # Run agent system (human review is disabled for web, we handle it later)
            result = run_agent_system(image_path, sensor_sequence, require_human_review=False)
            # Store result in session state
            st.session_state['result'] = result
            st.session_state['image_path'] = image_path
            st.session_state['sensor_sequence'] = sensor_sequence
            st.success("Analysis complete!")

# Display results if available
if 'result' in st.session_state:
    result = st.session_state['result']
    decision = result['decision']
    explanations = result['explanations']
    simulation = result.get('simulation', {})
    human_review = result.get('human_review', None)
    vision = result.get('vision', {})
    predictive = result.get('predictive', {})

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 AI Recommendation")
        st.markdown(f"**Action:** {decision['action']}")
        st.markdown(f"**Reasoning:** {decision['reasoning']}")
        st.markdown(f"**Confidence:** {decision['confidence']:.2f}")
        st.progress(decision['confidence'])

        st.subheader("📈 Prediction Details")
        st.metric("Defect Probability", f"{vision.get('defect_prob', 0):.2%}")
        st.metric("Failure Probability", f"{predictive.get('failure_prob', 0):.2%}")

    with col2:
        st.subheader("🧠 Explanations")
        st.text(explanations['summary'])

        if simulation:
            st.subheader("🎲 Digital Twin Simulation")
            df_sim = simulation['dataframe']
            st.dataframe(df_sim)
            st.info(f"Best scenario by cost: **{simulation['best_scenario']}**")

    # Human Decision Controls
    st.subheader("👤 Human Supervisor Review")
    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("✅ Approve", key="approve"):
            human_record = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'ai_recommendation': decision['action'],
                'ai_reasoning': decision['reasoning'],
                'ai_confidence': decision['confidence'],
                'human_status': 'approved',
                'modified_action': None,
                'feedback': ''
            }
            st.session_state['human_record'] = human_record
            st.success("Approved!")
    with colB:
        if st.button("❌ Reject", key="reject"):
            feedback = st.text_input("Reason for rejection", key="reject_feedback")
            if st.button("Submit Rejection"):
                human_record = {
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'ai_recommendation': decision['action'],
                    'ai_reasoning': decision['reasoning'],
                    'ai_confidence': decision['confidence'],
                    'human_status': 'rejected',
                    'modified_action': None,
                    'feedback': feedback
                }
                st.session_state['human_record'] = human_record
                st.success("Rejected!")
    with colC:
        if st.button("✏️ Modify", key="modify"):
            new_action = st.text_input("Enter modified action", key="modify_action")
            reason = st.text_input("Reason for modification", key="modify_reason")
            if st.button("Submit Modification"):
                human_record = {
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'ai_recommendation': decision['action'],
                    'ai_reasoning': decision['reasoning'],
                    'ai_confidence': decision['confidence'],
                    'human_status': 'modified',
                    'modified_action': new_action,
                    'feedback': reason
                }
                st.session_state['human_record'] = human_record
                st.success("Modified!")

    # Generate Report
    if 'human_record' in st.session_state:
        if st.button("📄 Generate PDF Report"):
            human_record = st.session_state['human_record']
            # Prepare report data
            report_data = {
                'decision': decision,
                'explanations': explanations,
                'vision': vision,
                'predictive': predictive,
                'simulation': simulation,
                'human_record': human_record,
                'image_path': st.session_state.get('image_path', None)
            }
            pdf_buffer = generate_report(report_data)
            st.download_button(
                label="Download Report",
                data=pdf_buffer,
                file_name=f"factory_decision_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )

    # Optionally display original image
    if 'image_path' in st.session_state:
        st.image(st.session_state['image_path'], caption="Analyzed Image", width=300)

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("AI Factory Intelligence Command Center v1.0")
# src/agent_system.py
import torch
import numpy as np
import pandas as pd
from PIL import Image
import joblib
from torchvision import transforms
import sys
import os
sys.path.append(os.path.dirname(__file__))

# Import our modules
from vision_advanced import load_resnet_model, generate_gradcam, estimate_severity
from rag_pipeline import RAGKnowledgeBase

# --------------------------------------------
# 1. Agent Base Class
# --------------------------------------------
class Agent:
    def __init__(self, name):
        self.name = name

    def process(self, shared_state):
        raise NotImplementedError

# --------------------------------------------
# 2. Vision Agent
# --------------------------------------------
class VisionAgent(Agent):
    def __init__(self, model_path='models/resnet_defect.pth'):
        super().__init__('VisionAgent')
        self.model = load_resnet_model(model_path)
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ])

    def process(self, shared_state):
        image_path = shared_state.get('image_path')
        if not image_path:
            raise ValueError("No image_path in shared_state")
        img = Image.open(image_path).convert('L')
        img_tensor = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            prob = torch.sigmoid(self.model(img_tensor)).item()
        is_defect = prob > 0.5
        severity = estimate_severity(image_path)
        shared_state['vision'] = {
            'defect_prob': prob,
            'is_defective': is_defect,
            'severity_ratio': severity,
        }
        print(f"[VisionAgent] Defect prob: {prob:.2f}, Severity: {severity:.2%}")
        return shared_state

# --------------------------------------------
# 3. Predictive Maintenance Agent (with fallback)
# --------------------------------------------
class PredictiveMaintenanceAgent(Agent):
    def __init__(self, model_path='models/timeseries_model.pkl', scaler_path='models/ts_scaler.pkl'):
        super().__init__('PredictiveMaintenanceAgent')
        self.lookback = 24
        self.feature_cols = ['vibration', 'pressure', 'rpm']

        # Try to load real model and scaler
        model_abs = os.path.abspath(model_path)
        scaler_abs = os.path.abspath(scaler_path)
        print(f"Looking for time-series model at: {model_abs}")

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print("[PredictiveMaintenanceAgent] Loaded real model and scaler.")
        else:
            print("[PredictiveMaintenanceAgent] Warning: Model files not found. Creating dummy model and scaler.")
            # Create dummy scaler (identity)
            self.scaler = StandardScaler()
            self.scaler.mean_ = np.zeros(3)
            self.scaler.scale_ = np.ones(3)
            self.scaler.n_features_in_ = 3
            # Create a dummy model that always predicts 0.5
            class DummyModel:
                def predict_proba(self, X):
                    return np.ones((X.shape[0], 2)) * 0.5
            self.model = DummyModel()

    def process(self, shared_state):
        sensor_data = shared_state.get('sensor_sequence')
        if sensor_data is None:
            raise ValueError("No sensor_sequence in shared_state")
        df = pd.DataFrame(sensor_data)[self.feature_cols]
        scaled = self.scaler.transform(df.values)
        if len(scaled) < self.lookback:
            pad = np.zeros((self.lookback - len(scaled), len(self.feature_cols)))
            seq = np.vstack([pad, scaled])
        else:
            seq = scaled[-self.lookback:]
        seq_flat = seq.flatten().reshape(1, -1)
        failure_prob = self.model.predict_proba(seq_flat)[0, 1]
        shared_state['predictive'] = {
            'failure_prob': float(failure_prob),
            'time_to_failure_hours': 2.5
        }
        print(f"[PredictiveMaintenanceAgent] Failure prob: {failure_prob:.2f}")
        return shared_state

# --------------------------------------------
# 4. Knowledge Agent (RAG)
# --------------------------------------------
class KnowledgeAgent(Agent):
    def __init__(self, pdf_paths=['data/raw/M1_manual.pdf', 'data/raw/SOP_general.pdf']):
        super().__init__('KnowledgeAgent')
        self.rag = RAGKnowledgeBase(pdf_paths)

    def process(self, shared_state):
        vision = shared_state.get('vision', {})
        predictive = shared_state.get('predictive', {})
        query = f"Machine has defect probability {vision.get('defect_prob', 0):.2f} and failure probability {predictive.get('failure_prob', 0):.2f}. What procedures apply?"
        retrieved = self.rag.retrieve(query, top_k=2)
        evidence_list = [r['text'] for r in retrieved]
        sources = [r['source'] for r in retrieved]
        shared_state['knowledge'] = {
            'evidence': evidence_list,
            'sources': sources,
            'retrieved_chunks': retrieved
        }
        print(f"[KnowledgeAgent] Retrieved {len(evidence_list)} evidence chunks.")
        return shared_state

# --------------------------------------------
# 5. Planning/Decision Agent
# --------------------------------------------
class PlanningAgent(Agent):
    def __init__(self):
        super().__init__('PlanningAgent')

    def process(self, shared_state):
        vision = shared_state.get('vision', {})
        predictive = shared_state.get('predictive', {})
        knowledge = shared_state.get('knowledge', {})
        defect_prob = vision.get('defect_prob', 0)
        fail_prob = predictive.get('failure_prob', 0)
        action = "Continue normal operation"
        reasoning = "No critical issues detected."
        confidence = 0.9

        if defect_prob > 0.8 and fail_prob > 0.7:
            action = "Stop machine immediately and schedule maintenance"
            reasoning = "High defect probability and high failure risk. Manual indicates overheating may cause both."
            confidence = 0.95
        elif defect_prob > 0.7:
            action = "Inspect product and adjust quality control parameters"
            reasoning = "Product defects are increasing; check camera calibration and production speed."
            confidence = 0.85
        elif fail_prob > 0.8:
            action = "Reduce production speed by 20% and monitor vibration"
            reasoning = "Failure risk elevated; vibration patterns suggest bearing wear per SOP."
            confidence = 0.80
        else:
            if knowledge.get('evidence'):
                ev = knowledge['evidence'][0]
                if 'reduce speed' in ev.lower():
                    action = "Reduce speed as per manual"
                    reasoning = ev[:100] + "..."
                    confidence = 0.75
        shared_state['decision'] = {
            'action': action,
            'reasoning': reasoning,
            'confidence': confidence
        }
        print(f"[PlanningAgent] Decision: {action}")
        return shared_state

# --------------------------------------------
# 6. Orchestrator
# --------------------------------------------
def run_agent_system(image_path, sensor_sequence):
    shared_state = {
        'image_path': image_path,
        'sensor_sequence': sensor_sequence
    }
    agents = [
        VisionAgent(),
        PredictiveMaintenanceAgent(),
        KnowledgeAgent(),
        PlanningAgent()
    ]
    for agent in agents:
        shared_state = agent.process(shared_state)
    return shared_state['decision']

# --------------------------------------------
# 7. Demo
# --------------------------------------------
if __name__ == "__main__":
    import random
    from sklearn.preprocessing import StandardScaler  # for dummy

    img_files = [f for f in os.listdir('data/images') if f.endswith('.png')]
    if img_files:
        image_path = os.path.join('data/images', random.choice(img_files))
    else:
        image_path = 'data/images/prod_0.png'

    sensor_seq = []
    for i in range(24):
        sensor_seq.append({
            'vibration': 0.5 + 0.1 * i + np.random.normal(0, 0.1),
            'pressure': 100 + 2 * np.sin(i/3) + np.random.normal(0, 1),
            'rpm': 3000 - 10 * i + np.random.normal(0, 20)
        })

    print("Running Agent System with image:", image_path)
    decision = run_agent_system(image_path, sensor_seq)
    print("\n🏭 Final Recommendation:")
    print(f"Action: {decision['action']}")
    print(f"Reasoning: {decision['reasoning']}")
    print(f"Confidence: {decision['confidence']:.2f}")


    # # src/agent_system.py
# import torch
# import numpy as np
# import pandas as pd
# from PIL import Image
# import joblib
# from torchvision import transforms
# import sys
# import os
# sys.path.append(os.path.dirname(__file__))

# # Import our modules (make sure these exist)
# from vision_advanced import load_resnet_model, generate_gradcam, estimate_severity
# from rag_pipeline import RAGKnowledgeBase

# # --------------------------------------------
# # 1. Agent Base Class
# # --------------------------------------------
# class Agent:
#     def __init__(self, name):
#         self.name = name

#     def process(self, shared_state):
#         raise NotImplementedError

# # --------------------------------------------
# # 2. Vision Agent
# # --------------------------------------------
# class VisionAgent(Agent):
#     def __init__(self, model_path='models/resnet_defect.pth'):
#         super().__init__('VisionAgent')
#         self.model = load_resnet_model(model_path)
#         self.transform = transforms.Compose([
#             transforms.Grayscale(num_output_channels=3),
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
#         ])

#     def process(self, shared_state):
#         image_path = shared_state.get('image_path')
#         if not image_path:
#             raise ValueError("No image_path in shared_state")
#         img = Image.open(image_path).convert('L')
#         img_tensor = self.transform(img).unsqueeze(0)
#         with torch.no_grad():
#             prob = torch.sigmoid(self.model(img_tensor)).item()
#         is_defect = prob > 0.5
#         severity = estimate_severity(image_path)
#         shared_state['vision'] = {
#             'defect_prob': prob,
#             'is_defective': is_defect,
#             'severity_ratio': severity,
#         }
#         print(f"[VisionAgent] Defect prob: {prob:.2f}, Severity: {severity:.2%}")
#         return shared_state

# # --------------------------------------------
# # 3. Predictive Maintenance Agent (sklearn version)
# # --------------------------------------------
# class PredictiveMaintenanceAgent(Agent):
#     def __init__(self, model_path='models/timeseries_model.pkl', scaler_path='models/ts_scaler.pkl'):
#         super().__init__('PredictiveMaintenanceAgent')
#         self.model = joblib.load(model_path)          # LogisticRegression
#         self.scaler = joblib.load(scaler_path)
#         self.lookback = 24
#         self.feature_cols = ['vibration', 'pressure', 'rpm']

#     def process(self, shared_state):
#         sensor_data = shared_state.get('sensor_sequence')
#         if sensor_data is None:
#             raise ValueError("No sensor_sequence in shared_state")
#         df = pd.DataFrame(sensor_data)[self.feature_cols]
#         scaled = self.scaler.transform(df.values)
#         if len(scaled) < self.lookback:
#             pad = np.zeros((self.lookback - len(scaled), len(self.feature_cols)))
#             seq = np.vstack([pad, scaled])
#         else:
#             seq = scaled[-self.lookback:]
#         seq_flat = seq.flatten().reshape(1, -1)
#         failure_prob = self.model.predict_proba(seq_flat)[0, 1]
#         shared_state['predictive'] = {
#             'failure_prob': float(failure_prob),
#             'time_to_failure_hours': 2.5   # dummy
#         }
#         print(f"[PredictiveMaintenanceAgent] Failure prob: {failure_prob:.2f}")
#         return shared_state

# # --------------------------------------------
# # 4. Knowledge Agent (RAG)
# # --------------------------------------------
# class KnowledgeAgent(Agent):
#     def __init__(self, pdf_paths=['data/raw/M1_manual.pdf', 'data/raw/SOP_general.pdf']):
#         super().__init__('KnowledgeAgent')
#         self.rag = RAGKnowledgeBase(pdf_paths)

#     def process(self, shared_state):
#         vision = shared_state.get('vision', {})
#         predictive = shared_state.get('predictive', {})
#         query = f"Machine has defect probability {vision.get('defect_prob', 0):.2f} and failure probability {predictive.get('failure_prob', 0):.2f}. What procedures apply?"
#         retrieved = self.rag.retrieve(query, top_k=2)
#         evidence_list = [r['text'] for r in retrieved]
#         sources = [r['source'] for r in retrieved]
#         shared_state['knowledge'] = {
#             'evidence': evidence_list,
#             'sources': sources,
#             'retrieved_chunks': retrieved
#         }
#         print(f"[KnowledgeAgent] Retrieved {len(evidence_list)} evidence chunks.")
#         return shared_state

# # --------------------------------------------
# # 5. Planning/Decision Agent
# # --------------------------------------------
# class PlanningAgent(Agent):
#     def __init__(self):
#         super().__init__('PlanningAgent')

#     def process(self, shared_state):
#         vision = shared_state.get('vision', {})
#         predictive = shared_state.get('predictive', {})
#         knowledge = shared_state.get('knowledge', {})
#         defect_prob = vision.get('defect_prob', 0)
#         fail_prob = predictive.get('failure_prob', 0)
#         action = "Continue normal operation"
#         reasoning = "No critical issues detected."
#         confidence = 0.9

#         if defect_prob > 0.8 and fail_prob > 0.7:
#             action = "Stop machine immediately and schedule maintenance"
#             reasoning = "High defect probability and high failure risk. Manual indicates overheating may cause both."
#             confidence = 0.95
#         elif defect_prob > 0.7:
#             action = "Inspect product and adjust quality control parameters"
#             reasoning = "Product defects are increasing; check camera calibration and production speed."
#             confidence = 0.85
#         elif fail_prob > 0.8:
#             action = "Reduce production speed by 20% and monitor vibration"
#             reasoning = "Failure risk elevated; vibration patterns suggest bearing wear per SOP."
#             confidence = 0.80
#         else:
#             if knowledge.get('evidence'):
#                 ev = knowledge['evidence'][0]
#                 if 'reduce speed' in ev.lower():
#                     action = "Reduce speed as per manual"
#                     reasoning = ev[:100] + "..."
#                     confidence = 0.75
#         shared_state['decision'] = {
#             'action': action,
#             'reasoning': reasoning,
#             'confidence': confidence
#         }
#         print(f"[PlanningAgent] Decision: {action}")
#         return shared_state

# # --------------------------------------------
# # 6. Orchestrator
# # --------------------------------------------
# def run_agent_system(image_path, sensor_sequence):
#     shared_state = {
#         'image_path': image_path,
#         'sensor_sequence': sensor_sequence
#     }
#     agents = [
#         VisionAgent(),
#         PredictiveMaintenanceAgent(),
#         KnowledgeAgent(),
#         PlanningAgent()
#     ]
#     for agent in agents:
#         shared_state = agent.process(shared_state)
#     return shared_state['decision']

# # --------------------------------------------
# # 7. Demo (run if script is executed directly)
# # --------------------------------------------
# if __name__ == "__main__":
#     import random
#     img_files = [f for f in os.listdir('data/images') if f.endswith('.png')]
#     if img_files:
#         image_path = os.path.join('data/images', random.choice(img_files))
#     else:
#         image_path = 'data/images/prod_0.png'

#     # Simulate a sensor sequence of 24 steps
#     sensor_seq = []
#     for i in range(24):
#         sensor_seq.append({
#             'vibration': 0.5 + 0.1 * i + np.random.normal(0, 0.1),
#             'pressure': 100 + 2 * np.sin(i/3) + np.random.normal(0, 1),
#             'rpm': 3000 - 10 * i + np.random.normal(0, 20)
#         })

#     print("Running Agent System with image:", image_path)
#     decision = run_agent_system(image_path, sensor_seq)
#     print("\n🏭 Final Recommendation:")
#     print(f"Action: {decision['action']}")
#     print(f"Reasoning: {decision['reasoning']}")
#     print(f"Confidence: {decision['confidence']:.2f}")
 
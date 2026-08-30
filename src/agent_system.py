# src/agent_system.py – final, with all fixes
import torch
import numpy as np
import pandas as pd
from PIL import Image
import joblib
from torchvision import transforms
import sys
import os
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(__file__))

from vision_advanced import load_resnet_model, estimate_severity
from rag_pipeline import RAGKnowledgeBase
from explainability import explain_tabular, explain_image, explain_timeseries
from digital_twin import compare_scenarios, FactorySimulator
from human_review import human_review

# ---------- Agent Base ----------
class Agent:
    def __init__(self, name):
        self.name = name

    def process(self, shared_state):
        raise NotImplementedError

# ---------- Vision Agent ----------
class VisionAgent(Agent):
    def __init__(self, model_path='models/resnet_defect.pth'):
        super().__init__('VisionAgent')
        self.model = load_resnet_model(model_path)
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def process(self, shared_state):
        image_path = shared_state.get('image_path')
        if not image_path:
            raise ValueError("No image_path")
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

# ---------- Predictive Maintenance ----------
class PredictiveMaintenanceAgent(Agent):
    def __init__(self, model_path='models/timeseries_model.pkl', scaler_path='models/ts_scaler.pkl'):
        super().__init__('PredictiveMaintenanceAgent')
        self.lookback = 24
        self.feature_cols = ['vibration', 'pressure', 'rpm']
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print("[PredictiveMaintenanceAgent] Loaded real model.")
        else:
            print("[PredictiveMaintenanceAgent] Using dummy model.")
            self.scaler = StandardScaler()
            self.scaler.mean_ = np.zeros(3)
            self.scaler.scale_ = np.ones(3)
            class DummyModel:
                def predict_proba(self, X):
                    return np.ones((X.shape[0], 2)) * 0.5
            self.model = DummyModel()

    def process(self, shared_state):
        sensor_data = shared_state.get('sensor_sequence')
        if sensor_data is None:
            raise ValueError("No sensor_sequence")
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

# ---------- Knowledge Agent ----------
class KnowledgeAgent(Agent):
    def __init__(self, pdf_paths=['data/raw/M1_manual.pdf', 'data/raw/SOP_general.pdf']):
        super().__init__('KnowledgeAgent')
        self.rag = RAGKnowledgeBase(pdf_paths)

    def process(self, shared_state):
        vision = shared_state.get('vision', {})
        predictive = shared_state.get('predictive', {})
        query = f"Defect prob {vision.get('defect_prob',0):.2f}, failure prob {predictive.get('failure_prob',0):.2f}. What procedures apply?"
        retrieved = self.rag.retrieve(query, top_k=2)
        shared_state['knowledge'] = {
            'evidence': [r['text'] for r in retrieved],
            'sources': [r['source'] for r in retrieved],
            'retrieved_chunks': retrieved
        }
        print(f"[KnowledgeAgent] Retrieved {len(retrieved)} chunks.")
        return shared_state

# ---------- Simulation Agent ----------
class SimulationAgent(Agent):
    def __init__(self):
        super().__init__('SimulationAgent')

    def process(self, shared_state):
        predictive = shared_state.get('predictive', {})
        fail_prob = predictive.get('failure_prob', 0.5)
        health = {f'M{i}': max(0.1, 1.0 - fail_prob * 0.8) for i in range(1, 5)}
        wear = {f'M{i}': min(0.9, fail_prob * 0.9) for i in range(1, 5)}
        current_state = {'health': health, 'wear': wear}

        df_sim = compare_scenarios(current_state, hours=24)
        best = df_sim.iloc[df_sim['Cost (per hr)'].idxmin()]['Scenario']
        shared_state['simulation'] = {
            'dataframe': df_sim,
            'best_scenario': best
        }
        print("[SimulationAgent] Simulation completed.")
        print(df_sim.to_string(index=False))
        return shared_state

# ---------- Planning Agent ----------
class PlanningAgent(Agent):
    def __init__(self):
        super().__init__('PlanningAgent')

    def process(self, shared_state):
        vision = shared_state.get('vision', {})
        predictive = shared_state.get('predictive', {})
        knowledge = shared_state.get('knowledge', {})
        simulation = shared_state.get('simulation', {})
        defect_prob = vision.get('defect_prob', 0)
        fail_prob = predictive.get('failure_prob', 0)
        action = "Continue normal operation"
        reasoning = "No critical issues."
        confidence = 0.9

        if defect_prob > 0.8 and fail_prob > 0.7:
            action = "Stop machine and schedule maintenance"
            reasoning = "High defect & failure risk."
            confidence = 0.95
        elif defect_prob > 0.7:
            action = "Inspect product and adjust quality parameters"
            reasoning = "Defects increasing."
            confidence = 0.85
        elif fail_prob > 0.8:
            action = "Reduce speed by 20% and monitor vibration"
            reasoning = "Failure risk elevated."
            confidence = 0.80
        elif knowledge.get('evidence'):
            ev = knowledge['evidence'][0]
            if 'reduce speed' in ev.lower():
                action = "Reduce speed as per manual"
                reasoning = ev[:100] + "..."
                confidence = 0.75

        if simulation:
            best = simulation.get('best_scenario')
            if best == 'stop' and 'stop' not in action.lower():
                action = "Stop machine (simulation recommends)"
                reasoning += " Simulation suggests stopping is cost-effective."
                confidence = 0.85
            elif best == 'reduce_load' and 'reduce' not in action.lower():
                action = "Reduce load (simulation recommends)"
                reasoning += " Simulation indicates load reduction minimizes cost."
                confidence = 0.80

        shared_state['decision'] = {
            'action': action,
            'reasoning': reasoning,
            'confidence': confidence
        }
        print(f"[PlanningAgent] Decision: {action}")
        return shared_state

# ---------- Explainability Agent ----------
class ExplainabilityAgent(Agent):
    def __init__(self):
        super().__init__('ExplainabilityAgent')
        self.tabular_model = None
        for m in ['models/xgb_model.pkl', 'models/rf_model.pkl']:
            if os.path.exists(m):
                self.tabular_model = joblib.load(m)
                break
        self.image_model = load_resnet_model() if os.path.exists('models/resnet_defect.pth') else None
        self.ts_model = joblib.load('models/timeseries_model.pkl') if os.path.exists('models/timeseries_model.pkl') else None
        self.scaler = joblib.load('models/ts_scaler.pkl') if os.path.exists('models/ts_scaler.pkl') else None
        self.tabular_feature_names = ['production_count', 'temperature', 'vibration', 'pressure', 'rpm',
                                      'vibration_rolling_mean_24', 'vibration_rolling_std_24',
                                      'pressure_rolling_mean_24', 'pressure_rolling_std_24',
                                      'rpm_rolling_mean_24', 'rpm_rolling_std_24',
                                      'vibration_lag1', 'vibration_lag24',
                                      'pressure_lag1', 'pressure_lag24',
                                      'rpm_lag1', 'rpm_lag24',
                                      'machine_id_M2', 'machine_id_M3', 'machine_id_M4',
                                      'operator_OpB', 'operator_OpC']
        self.ts_feature_names = [f"t{i}_{f}" for i in range(24) for f in ['vib', 'pres', 'rpm']]

    def process(self, shared_state):
        explanations = {}
        # Image
        img_path = shared_state.get('image_path')
        if img_path and os.path.exists(img_path) and self.image_model:
            heatmap, exp = explain_image(self.image_model, img_path)
            explanations['image_heatmap'] = heatmap
            explanations['image_explanation'] = exp
        else:
            explanations['image_explanation'] = 'No image or model.'

        # Tabular – fixed dtype error
        if self.tabular_model:
            try:
                test_df = pd.read_csv('data/processed/test.csv')
                # Ensure all features are numeric to avoid SHAP dtype error
                sample = test_df.iloc[0][self.tabular_feature_names].astype(float).values.reshape(1, -1)
                _, _, tab_exp = explain_tabular(self.tabular_model, sample, self.tabular_feature_names)
                explanations['tabular_explanation'] = tab_exp
            except Exception as e:
                explanations['tabular_explanation'] = f"Tabular error: {e}"
        else:
            explanations['tabular_explanation'] = "Tabular model not loaded."

        # Time-series
        sensor_data = shared_state.get('sensor_sequence')
        if sensor_data and self.ts_model and self.scaler:
            try:
                df = pd.DataFrame(sensor_data)[['vibration', 'pressure', 'rpm']]
                scaled = self.scaler.transform(df.values)
                seq_flat = scaled[-24:].flatten().reshape(1, -1)
                _, ts_exp = explain_timeseries(self.ts_model, seq_flat, None, self.ts_feature_names)
                explanations['timeseries_explanation'] = ts_exp
            except Exception as e:
                explanations['timeseries_explanation'] = f"TS error: {e}"
        else:
            explanations['timeseries_explanation'] = "No TS model or data."

        summary = "Decision Explanation:\n"
        summary += f"- Image: {explanations.get('image_explanation', 'N/A')}\n"
        summary += f"- Tabular: {explanations.get('tabular_explanation', 'N/A')}\n"
        summary += f"- Time-series: {explanations.get('timeseries_explanation', 'N/A')}\n"
        explanations['summary'] = summary
        shared_state['explanations'] = explanations
        print("[ExplainabilityAgent] Explanation generated.")
        return shared_state

# ---------- Orchestrator ----------
def run_agent_system(image_path, sensor_sequence, require_human_review=True):
    shared_state = {
        'image_path': image_path,
        'sensor_sequence': sensor_sequence
    }
    agents = [
        VisionAgent(),
        PredictiveMaintenanceAgent(),
        KnowledgeAgent(),
        SimulationAgent(),
        PlanningAgent(),
        ExplainabilityAgent()
    ]
    for agent in agents:
        shared_state = agent.process(shared_state)

    if require_human_review:
        decision = shared_state['decision']
        explanations = shared_state['explanations']
        human_record = human_review(decision, explanations)
        shared_state['human_review'] = human_record
        if human_record['status'] == 'modified':
            shared_state['decision']['action'] = human_record['modified_action']
            shared_state['decision']['human_modified'] = True

    return shared_state

# ---------- CLI Demo (Human Review DISABLED) ----------
if __name__ == "__main__":
    import random

    img_files = [f for f in os.listdir('data/images') if f.endswith('.png')]
    image_path = os.path.join('data/images', random.choice(img_files)) if img_files else 'data/images/prod_0.png'

    sensor_seq = [{'vibration': 0.5 + 0.1 * i + np.random.normal(0, 0.1),
                   'pressure': 100 + 2 * np.sin(i / 3) + np.random.normal(0, 1),
                   'rpm': 3000 - 10 * i + np.random.normal(0, 20)} for i in range(24)]

    print("Running Agent System with image:", image_path)
    # Disable human review for CLI demo (web app handles it)
    result = run_agent_system(image_path, sensor_seq, require_human_review=False)
    decision = result['decision']
    print("\n🏭 Final Recommendation:")
    print(f"Action: {decision['action']}")
    print(f"Reasoning: {decision['reasoning']}")
    print(f"Confidence: {decision['confidence']:.2f}")
    print("\n📖 Explanation Summary:")
    print(result['explanations']['summary'])
    if 'simulation' in result:
        print("\n📊 Simulation Results:")
        print(result['simulation']['dataframe'].to_string(index=False))

        # # src/agent_system.py – final, fully functional
# import torch
# import numpy as np
# import pandas as pd
# from PIL import Image
# import joblib
# from torchvision import transforms
# import sys
# import os
# from sklearn.preprocessing import StandardScaler

# sys.path.append(os.path.dirname(__file__))

# from vision_advanced import load_resnet_model, estimate_severity
# from rag_pipeline import RAGKnowledgeBase
# from explainability import explain_tabular, explain_image, explain_timeseries
# from digital_twin import compare_scenarios, FactorySimulator   # Digital Twin
# from human_review import human_review                         # Human‑in‑the‑Loop

# # ---------- Agent Base ----------
# class Agent:
#     def __init__(self, name):
#         self.name = name

#     def process(self, shared_state):
#         raise NotImplementedError

# # ---------- Vision Agent ----------
# class VisionAgent(Agent):
#     def __init__(self, model_path='models/resnet_defect.pth'):
#         super().__init__('VisionAgent')
#         self.model = load_resnet_model(model_path)
#         self.transform = transforms.Compose([
#             transforms.Grayscale(num_output_channels=3),
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#         ])

#     def process(self, shared_state):
#         image_path = shared_state.get('image_path')
#         if not image_path:
#             raise ValueError("No image_path")
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

# # ---------- Predictive Maintenance (with fallback) ----------
# class PredictiveMaintenanceAgent(Agent):
#     def __init__(self, model_path='models/timeseries_model.pkl', scaler_path='models/ts_scaler.pkl'):
#         super().__init__('PredictiveMaintenanceAgent')
#         self.lookback = 24
#         self.feature_cols = ['vibration', 'pressure', 'rpm']
#         if os.path.exists(model_path) and os.path.exists(scaler_path):
#             self.model = joblib.load(model_path)
#             self.scaler = joblib.load(scaler_path)
#             print("[PredictiveMaintenanceAgent] Loaded real model.")
#         else:
#             print("[PredictiveMaintenanceAgent] Using dummy model.")
#             self.scaler = StandardScaler()
#             self.scaler.mean_ = np.zeros(3)
#             self.scaler.scale_ = np.ones(3)
#             class DummyModel:
#                 def predict_proba(self, X):
#                     return np.ones((X.shape[0], 2)) * 0.5
#             self.model = DummyModel()

#     def process(self, shared_state):
#         sensor_data = shared_state.get('sensor_sequence')
#         if sensor_data is None:
#             raise ValueError("No sensor_sequence")
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
#             'time_to_failure_hours': 2.5
#         }
#         print(f"[PredictiveMaintenanceAgent] Failure prob: {failure_prob:.2f}")
#         return shared_state

# # ---------- Knowledge Agent (RAG) ----------
# class KnowledgeAgent(Agent):
#     def __init__(self, pdf_paths=['data/raw/M1_manual.pdf', 'data/raw/SOP_general.pdf']):
#         super().__init__('KnowledgeAgent')
#         self.rag = RAGKnowledgeBase(pdf_paths)

#     def process(self, shared_state):
#         vision = shared_state.get('vision', {})
#         predictive = shared_state.get('predictive', {})
#         query = f"Defect prob {vision.get('defect_prob',0):.2f}, failure prob {predictive.get('failure_prob',0):.2f}. What procedures apply?"
#         retrieved = self.rag.retrieve(query, top_k=2)
#         shared_state['knowledge'] = {
#             'evidence': [r['text'] for r in retrieved],
#             'sources': [r['source'] for r in retrieved],
#             'retrieved_chunks': retrieved
#         }
#         print(f"[KnowledgeAgent] Retrieved {len(retrieved)} chunks.")
#         return shared_state

# # ---------- Simulation Agent (Digital Twin) ----------
# class SimulationAgent(Agent):
#     def __init__(self):
#         super().__init__('SimulationAgent')

#     def process(self, shared_state):
#         predictive = shared_state.get('predictive', {})
#         fail_prob = predictive.get('failure_prob', 0.5)
#         # Map failure prob to health/wear
#         health = {f'M{i}': max(0.1, 1.0 - fail_prob * 0.8) for i in range(1, 5)}
#         wear = {f'M{i}': min(0.9, fail_prob * 0.9) for i in range(1, 5)}
#         current_state = {'health': health, 'wear': wear}

#         df_sim = compare_scenarios(current_state, hours=24)
#         best = df_sim.iloc[df_sim['Cost (per hr)'].idxmin()]['Scenario']
#         shared_state['simulation'] = {
#             'dataframe': df_sim,
#             'best_scenario': best
#         }
#         print("[SimulationAgent] Simulation completed.")
#         print(df_sim.to_string(index=False))
#         return shared_state

# # ---------- Planning Agent (incorporates simulation) ----------
# class PlanningAgent(Agent):
#     def __init__(self):
#         super().__init__('PlanningAgent')

#     def process(self, shared_state):
#         vision = shared_state.get('vision', {})
#         predictive = shared_state.get('predictive', {})
#         knowledge = shared_state.get('knowledge', {})
#         simulation = shared_state.get('simulation', {})
#         defect_prob = vision.get('defect_prob', 0)
#         fail_prob = predictive.get('failure_prob', 0)
#         action = "Continue normal operation"
#         reasoning = "No critical issues."
#         confidence = 0.9

#         if defect_prob > 0.8 and fail_prob > 0.7:
#             action = "Stop machine and schedule maintenance"
#             reasoning = "High defect & failure risk."
#             confidence = 0.95
#         elif defect_prob > 0.7:
#             action = "Inspect product and adjust quality parameters"
#             reasoning = "Defects increasing."
#             confidence = 0.85
#         elif fail_prob > 0.8:
#             action = "Reduce speed by 20% and monitor vibration"
#             reasoning = "Failure risk elevated."
#             confidence = 0.80
#         elif knowledge.get('evidence'):
#             ev = knowledge['evidence'][0]
#             if 'reduce speed' in ev.lower():
#                 action = "Reduce speed as per manual"
#                 reasoning = ev[:100] + "..."
#                 confidence = 0.75

#         # Incorporate simulation best scenario
#         if simulation:
#             best = simulation.get('best_scenario')
#             if best == 'stop' and 'stop' not in action.lower():
#                 action = "Stop machine (simulation recommends)"
#                 reasoning += " Simulation suggests stopping is cost-effective."
#                 confidence = 0.85
#             elif best == 'reduce_load' and 'reduce' not in action.lower():
#                 action = "Reduce load (simulation recommends)"
#                 reasoning += " Simulation indicates load reduction minimizes cost."
#                 confidence = 0.80

#         shared_state['decision'] = {
#             'action': action,
#             'reasoning': reasoning,
#             'confidence': confidence
#         }
#         print(f"[PlanningAgent] Decision: {action}")
#         return shared_state

# # ---------- Explainability Agent ----------
# class ExplainabilityAgent(Agent):
#     def __init__(self):
#         super().__init__('ExplainabilityAgent')
#         # Load models (with fallback)
#         self.tabular_model = None
#         for m in ['models/xgb_model.pkl', 'models/rf_model.pkl']:
#             if os.path.exists(m):
#                 self.tabular_model = joblib.load(m)
#                 break
#         self.image_model = load_resnet_model() if os.path.exists('models/resnet_defect.pth') else None
#         self.ts_model = joblib.load('models/timeseries_model.pkl') if os.path.exists('models/timeseries_model.pkl') else None
#         self.scaler = joblib.load('models/ts_scaler.pkl') if os.path.exists('models/ts_scaler.pkl') else None
#         self.tabular_feature_names = ['production_count', 'temperature', 'vibration', 'pressure', 'rpm',
#                                       'vibration_rolling_mean_24', 'vibration_rolling_std_24',
#                                       'pressure_rolling_mean_24', 'pressure_rolling_std_24',
#                                       'rpm_rolling_mean_24', 'rpm_rolling_std_24',
#                                       'vibration_lag1', 'vibration_lag24',
#                                       'pressure_lag1', 'pressure_lag24',
#                                       'rpm_lag1', 'rpm_lag24',
#                                       'machine_id_M2', 'machine_id_M3', 'machine_id_M4',
#                                       'operator_OpB', 'operator_OpC']
#         self.ts_feature_names = [f"t{i}_{f}" for i in range(24) for f in ['vib', 'pres', 'rpm']]

#     def process(self, shared_state):
#         explanations = {}
#         # Image
#         img_path = shared_state.get('image_path')
#         if img_path and os.path.exists(img_path) and self.image_model:
#             heatmap, exp = explain_image(self.image_model, img_path)
#             explanations['image_heatmap'] = heatmap
#             explanations['image_explanation'] = exp
#         else:
#             explanations['image_explanation'] = 'No image or model.'

#         # Tabular
#         if self.tabular_model:
#             try:
#                 test_df = pd.read_csv('data/processed/test.csv')
#                 sample = test_df.iloc[0][self.tabular_feature_names].values.reshape(1, -1)
#                 _, _, tab_exp = explain_tabular(self.tabular_model, sample, self.tabular_feature_names)
#                 explanations['tabular_explanation'] = tab_exp
#             except Exception as e:
#                 explanations['tabular_explanation'] = f"Tabular error: {e}"
#         else:
#             explanations['tabular_explanation'] = "Tabular model not loaded."

#         # Time-series
#         sensor_data = shared_state.get('sensor_sequence')
#         if sensor_data and self.ts_model and self.scaler:
#             try:
#                 df = pd.DataFrame(sensor_data)[['vibration', 'pressure', 'rpm']]
#                 scaled = self.scaler.transform(df.values)
#                 seq_flat = scaled[-24:].flatten().reshape(1, -1)
#                 _, ts_exp = explain_timeseries(self.ts_model, seq_flat, None, self.ts_feature_names)
#                 explanations['timeseries_explanation'] = ts_exp
#             except Exception as e:
#                 explanations['timeseries_explanation'] = f"TS error: {e}"
#         else:
#             explanations['timeseries_explanation'] = "No TS model or data."

#         # Summary
#         summary = "Decision Explanation:\n"
#         summary += f"- Image: {explanations.get('image_explanation', 'N/A')}\n"
#         summary += f"- Tabular: {explanations.get('tabular_explanation', 'N/A')}\n"
#         summary += f"- Time-series: {explanations.get('timeseries_explanation', 'N/A')}\n"
#         explanations['summary'] = summary
#         shared_state['explanations'] = explanations
#         print("[ExplainabilityAgent] Explanation generated.")
#         return shared_state

# # ---------- Orchestrator ----------
# def run_agent_system(image_path, sensor_sequence, require_human_review=True):
#     shared_state = {
#         'image_path': image_path,
#         'sensor_sequence': sensor_sequence
#     }
#     agents = [
#         VisionAgent(),
#         PredictiveMaintenanceAgent(),
#         KnowledgeAgent(),
#         SimulationAgent(),
#         PlanningAgent(),
#         ExplainabilityAgent()
#     ]
#     for agent in agents:
#         shared_state = agent.process(shared_state)

#     # Human-in-the-Loop
#     if require_human_review:
#         decision = shared_state['decision']
#         explanations = shared_state['explanations']
#         human_record = human_review(decision, explanations)
#         shared_state['human_review'] = human_record
#         if human_record['status'] == 'modified':
#             shared_state['decision']['action'] = human_record['modified_action']
#             shared_state['decision']['human_modified'] = True

#     return shared_state

# # ---------- Demo ----------
# if __name__ == "__main__":
#     import random

#     img_files = [f for f in os.listdir('data/images') if f.endswith('.png')]
#     image_path = os.path.join('data/images', random.choice(img_files)) if img_files else 'data/images/prod_0.png'

#     sensor_seq = [{'vibration': 0.5 + 0.1 * i + np.random.normal(0, 0.1),
#                    'pressure': 100 + 2 * np.sin(i / 3) + np.random.normal(0, 1),
#                    'rpm': 3000 - 10 * i + np.random.normal(0, 20)} for i in range(24)]

#     print("Running Agent System with image:", image_path)
#     result = run_agent_system(image_path, sensor_seq, require_human_review=True)
#     decision = result['decision']
#     print("\n🏭 Final Recommendation:")
#     print(f"Action: {decision['action']}")
#     print(f"Reasoning: {decision['reasoning']}")
#     print(f"Confidence: {decision['confidence']:.2f}")
#     print("\n📖 Explanation Summary:")
#     print(result['explanations']['summary'])
#     if 'human_review' in result:
#         print("\n👤 Human Review Record:")
#         print(result['human_review'])
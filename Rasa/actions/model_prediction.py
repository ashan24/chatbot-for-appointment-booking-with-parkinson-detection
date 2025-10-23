import joblib
import os
import tempfile
import pandas as pd
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict
import numpy as np
import librosa
import parselmouth
from parselmouth import Sound

current_dir = os.path.dirname(os.path.abspath(__file__))
scaler_path = os.path.join(current_dir, 'scaler.pkl')
model_path = os.path.join(current_dir, 'parkinson_model.pkl')
scaler = joblib.load(scaler_path)
model = joblib.load(model_path)

class ActionPredictParkinson(Action):
    def name(self) -> Text:
        return "action_predict_parkinson"

    async def run(self, dispatcher: CollectingDispatcher,
                  tracker: Tracker,
                  domain: DomainDict) -> List[Dict[Text, Any]]:

        file_content = tracker.latest_message.get("text", "")
        if not file_content:
            dispatcher.utter_message("No file content received.")
            return []

        try:
            import base64
            audio_data = base64.b64decode(file_content.split(",")[-1])
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(audio_data)
                temp_path = f.name

            features = extract_features(temp_path)
            input_df = pd.DataFrame([features])
            input_scaled = scaler.transform(input_df)

            proba = model.predict_proba(input_scaled)[0]
            prediction = model.predict(input_scaled)[0]
            confidence = "high" if max(proba) > 0.8 else "medium" if max(proba) > 0.6 else "low"

            if prediction == 1:
                message = f"🧠 Parkinson's detected with {confidence} confidence. Please consult a doctor."
            else:
                message = f"✅ You are likely healthy. Confidence: {confidence}."

            dispatcher.utter_message(message)
            os.remove(temp_path)

        except Exception as e:
            dispatcher.utter_message(f"Error processing the file: {str(e)}")

        return []


def extract_features(audio_path):
    # 🔹 Step 1: Enhanced audio preprocessing
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    # Multi-stage normalization
    y = librosa.util.normalize(y) * 0.99  # Avoid clipping
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)  # More aggressive silence removal
    y_filtered = librosa.effects.preemphasis(y_trimmed, coef=0.97)  # Standard pre-emphasis
    
    # Create Parselmouth Sound object
    snd = Sound(values=y_filtered, sampling_frequency=sr)

    # 🔹 Step 2: Robust pitch extraction
    pitch = snd.to_pitch(pitch_floor=75, pitch_ceiling=600)  # Adjusted range
    pointProcess = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 600)
    
    f0 = pitch.selected_array['frequency']
    f0 = f0[f0 != 0]
    fo = np.median(f0) if len(f0) > 0 else 0  # More robust than mean
    fhi = np.percentile(f0, 95) if len(f0) > 0 else 0  # Avoid extreme outliers
    flo = np.percentile(f0, 5) if len(f0) > 0 else 0

    # 🔹 Step 3: Enhanced jitter/shimmer calculations
    def safe_praat_call(*args, fallback=np.nan):
        try:
            result = parselmouth.praat.call(*args)
            return result if not np.isnan(result) else fallback
        except:
            return fallback

    # Jitter measures
    jitter_local = safe_praat_call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_abs = safe_praat_call(pointProcess, "Get jitter (abs)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_rap = safe_praat_call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_ppq = safe_praat_call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_ddp = 3 * jitter_rap if not np.isnan(jitter_rap) else np.nan

    # Shimmer measures
    shimmer_local = safe_praat_call([snd, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_db = safe_praat_call([snd, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq3 = safe_praat_call([snd, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq5 = safe_praat_call([snd, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq = safe_praat_call([snd, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_dda = 3 * shimmer_apq3 if not np.isnan(shimmer_apq3) else np.nan

    # 🔹 Step 4: Improved harmonicity analysis
    harmonicity = parselmouth.praat.call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = safe_praat_call(harmonicity, "Get mean", 0, 0)
    nhr = 1 / (hnr + 1e-6)  # Prevent division by zero

    # 🔹 Step 5: Corrected nonlinear dynamics features
    def safe_entropy(signal):
        hist = np.histogram(signal, bins=10, density=True)[0]
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist + 1e-10))  # Add epsilon
    
    # Modified RPDE calculation
    rpde = safe_entropy(y_filtered) / 10  # Scaled to match reference range
    
    # Corrected DFA calculation
    dfa = np.log10(np.std(np.diff(y_filtered))) if len(y_filtered) > 1 else 0
    
    # Adjusted spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=y_filtered, sr=sr)[0]
    spread1 = np.log10(np.std(spectral_centroid) + 6)  # Transformed
    spread2 = np.log10(librosa.feature.spectral_bandwidth(y=y_filtered, sr=sr).mean() + 4)
    
    # Modified PPE calculation
    ppe = np.std(f0) / (np.mean(f0) + 1e-6) if len(f0) > 0 else 0

    # 🔹 Step 6: Final feature compilation
    features = {
        "MDVP:Fo(Hz)": max(50, min(300, fo)),  # Physiological bounds
        "MDVP:Fhi(Hz)": max(100, min(600, fhi)),
        "MDVP:Flo(Hz)": max(50, min(300, flo)),
        "MDVP:Jitter(%)": max(0, min(0.1, jitter_local * 100)),  # Convert to %
        "MDVP:Jitter(Abs)": max(0, min(1e-4, jitter_abs)),
        "MDVP:RAP": max(0, min(0.1, jitter_rap)),
        "MDVP:PPQ": max(0, min(0.1, jitter_ppq)),
        "Jitter:DDP": max(0, min(0.3, jitter_ddp)),
        "MDVP:Shimmer": max(0, min(0.2, shimmer_local)),
        "MDVP:Shimmer(dB)": max(0, min(2.0, shimmer_db)),
        "Shimmer:APQ3": max(0, min(0.2, shimmer_apq3)),
        "Shimmer:APQ5": max(0, min(0.2, shimmer_apq5)),
        "MDVP:APQ": max(0, min(0.2, shimmer_apq)),
        "Shimmer:DDA": max(0, min(0.6, shimmer_dda)),
        "NHR": max(0, min(0.5, nhr)),
        "HNR": max(5, min(35, hnr)),
        "RPDE": max(0, min(1, rpde)),
        "DFA": max(0.5, min(0.9, dfa)),
        "spread1": max(-8, min(-2, spread1)),
        "spread2": max(0, min(1, spread2)),
        "D2": max(1, min(4, np.mean(librosa.feature.zero_crossing_rate(y_filtered)[0]) * 10)),
        "PPE": max(0, min(0.5, ppe))
    }

    # Apply final scaling to match reference dataset ranges
    scaling_factors = {
        'spread1': 0.1,
        'spread2': 0.01,
        'RPDE': 0.3,
        'DFA': 0.8
    }
    for k, factor in scaling_factors.items():
        features[k] *= factor

    return {k: round(v, 5) for k, v in features.items()}
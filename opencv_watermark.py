"""
Adaptive OpenCV Watermark Detector (ORB Feature Matching)
---------------------------------------------------------
Advanced fallback system that "learns" from Gemini.
Uses ORB (Oriented FAST and Rotated BRIEF) for robust, invariant detection.
"""

import cv2
import numpy as np
import os
import logging
import uuid
import glob
import pickle
import csv

# Try to import ML libraries
try:
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    HAS_ML = True
except ImportError:
    HAS_ML = False

logger = logging.getLogger("opencv_watermark")

TEMPLATE_DIR = "watermark_templates"
POSITIVE_DIR = os.path.join(TEMPLATE_DIR, "positive")
NEGATIVE_DIR = os.path.join(TEMPLATE_DIR, "negative")
DATASET_FILE = "watermark_dataset.csv"
MODEL_FILE = "watermark_model.pkl"

os.makedirs(POSITIVE_DIR, exist_ok=True)
os.makedirs(NEGATIVE_DIR, exist_ok=True)

class WatermarkLearner:
    @staticmethod
    def extract_features(frame: np.ndarray, coords: dict, match_data: dict = None) -> dict:
        """
        Extract numerical features for ML training.
        """
        try:
            x, y, w, h = coords['x'], coords['y'], coords['w'], coords['h']
            h_img, w_img = frame.shape[:2]
            
            # Geometric Features
            rel_x = x / w_img
            rel_y = y / h_img
            rel_w = w / w_img
            rel_h = h / h_img
            aspect_ratio = w / h if h > 0 else 0
            
            # Visual Features (Texture/Complexity)
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0: return None
            
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            std_dev = np.std(gray_roi)
            edge_density = np.mean(cv2.Canny(gray_roi, 100, 200)) / 255.0
            
            features = {
                "rel_x": rel_x, "rel_y": rel_y, "rel_w": rel_w, "rel_h": rel_h,
                "aspect_ratio": aspect_ratio,
                "std_dev": std_dev, "edge_density": edge_density,
                "confidence": match_data.get("confidence", 0) if match_data else 0,
                "matches": match_data.get("matches", 0) if match_data else 0
            }
            return features
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return None

    @staticmethod
    def log_feedback(features: dict, label: int):
        """
        Log features and label (1=Watermark, 0=False Positive) to CSV.
        """
        if not features: return
        
        file_exists = os.path.exists(DATASET_FILE)
        
        try:
            with open(DATASET_FILE, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(features.keys()) + ["label"])
                if not file_exists:
                    writer.writeheader()
                
                row = features.copy()
                row["label"] = label
                writer.writerow(row)
                
            logger.info(f"📝 Logged feedback to dataset (Label: {label})")
            
            # Trigger training if enough data
            if HAS_ML:
                WatermarkLearner.train_classifier()
                
        except Exception as e:
            logger.error(f"Failed to log feedback: {e}")

    @staticmethod
    def train_classifier():
        """
        Train Random Forest Classifier if enough data exists.
        """
        if not HAS_ML: return
        
        try:
            df = pd.read_csv(DATASET_FILE)
            if len(df) < 10: return # Need minimum data
            
            # Check class balance
            if len(df['label'].unique()) < 2: return # Need both classes
            
            X = df.drop(columns=['label'])
            y = df['label']
            
            model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            model.fit(X, y)
            
            with open(MODEL_FILE, 'wb') as f:
                pickle.dump(model, f)
                
            acc = model.score(X, y)
            logger.info(f"🧠 Trained Watermark Classifier (Accuracy: {acc:.2f}, Samples: {len(df)})")
            
        except Exception as e:
            logger.warning(f"Model training failed: {e}")

    @staticmethod
    def save_template(frame: np.ndarray, coords: dict, is_positive: bool = True):
        """
        Save a detection as a template AND log features.
        """
        try:
            x, y, w, h = coords['x'], coords['y'], coords['w'], coords['h']
            
            # Validation
            if w < 20 or h < 20: return
            
            # 1. Save Template Image
            roi = frame[y:y+h, x:x+w]
            label_str = "pos" if is_positive else "neg"
            target_dir = POSITIVE_DIR if is_positive else NEGATIVE_DIR
            
            filename = f"{label_str}_{uuid.uuid4().hex[:8]}.png"
            path = os.path.join(target_dir, filename)
            cv2.imwrite(path, roi)
            
            # 2. Log Features for ML
            # For initial learning, we assume high confidence/matches if it came from Gemini
            # But better to extract real ORB features if possible. 
            # For now, we extract visual/geometric features.
            features = WatermarkLearner.extract_features(frame, coords, match_data={"confidence": 1.0, "matches": 50})
            if features:
                WatermarkLearner.log_feedback(features, 1 if is_positive else 0)
            
            # 3. Update ORB Database
            WatermarkDetector.train()
            
            logger.info(f"🧠 Learned new {'POSITIVE' if is_positive else 'NEGATIVE'} template: {filename}")
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to learn template: {e}")

class WatermarkDetector:
    _orb = None
    _bf = None
    _positive_features = [] 
    _negative_features = []
    _model = None
    
    @classmethod
    def init(cls):
        if cls._orb is None:
            cls._orb = cv2.ORB_create(nfeatures=1000)
            cls._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            cls.load_features()
            cls.load_model()

    @classmethod
    def load_model(cls):
        if os.path.exists(MODEL_FILE):
            try:
                with open(MODEL_FILE, 'rb') as f:
                    cls._model = pickle.load(f)
                logger.info("🤖 Loaded ML Watermark Classifier")
            except: pass

    @classmethod
    def train(cls):
        """
        Re-build feature database from all templates (Positive & Negative).
        """
        cls.init()
        
        def load_from_dir(directory):
            feats = []
            templates = glob.glob(os.path.join(directory, "*.png"))
            for t_path in templates:
                img = cv2.imread(t_path, cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                
                kp, des = cls._orb.detectAndCompute(img, None)
                if des is not None and len(kp) > 5:
                    feats.append((kp, des, img.shape))
            return feats

        cls._positive_features = load_from_dir(POSITIVE_DIR)
        cls._negative_features = load_from_dir(NEGATIVE_DIR)
        
        logger.info(f"📚 Trained on {len(cls._positive_features)} POSITIVE and {len(cls._negative_features)} NEGATIVE templates.")

    @classmethod
    def load_features(cls):
        cls.train()

    @staticmethod
    def detect(frame: np.ndarray) -> dict:
        """
        Detect watermark using ORB + ML Validation.
        """
        WatermarkDetector.init()
        if not WatermarkDetector._positive_features:
            return None
            
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = WatermarkDetector._orb.detectAndCompute(gray_frame, None)
        
        if des_frame is None: return None
        
        best_match = None
        max_good_matches = 0
        
        # 1. Find Best Positive Match
        for (kp_template, des_template, t_shape) in WatermarkDetector._positive_features:
            if des_template is None: continue
            
            matches = WatermarkDetector._bf.match(des_template, des_frame)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = [m for m in matches if m.distance < 50]
            
            if len(good_matches) > max_good_matches:
                max_good_matches = len(good_matches)
                
                # Increased threshold from 4 to 10 for initial homography
                if len(good_matches) > 10:
                    src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    
                    if M is not None:
                        h, w = t_shape[:2]
                        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                        dst = cv2.perspectiveTransform(pts, M)
                        
                        x_min = int(np.min(dst[:, :, 0]))
                        y_min = int(np.min(dst[:, :, 1]))
                        x_max = int(np.max(dst[:, :, 0]))
                        y_max = int(np.max(dst[:, :, 1]))
                        
                        w_rect = x_max - x_min
                        h_rect = y_max - y_min
                        
                        # Size Check: Must be reasonable size (prevent tiny 8x57 noise)
                        if x_min >= 0 and y_min >= 0 and x_max < frame.shape[1] and y_max < frame.shape[0]:
                             if w_rect > 30 and h_rect > 15:
                                 best_match = {
                                    'x': x_min, 'y': y_min, 'w': w_rect, 'h': h_rect,
                                    'confidence': len(good_matches) / len(kp_template)
                                }

        # 2. Check against Negative Templates (False Positive Rejection)
        # Increased threshold from 8 to 12
        if best_match and max_good_matches > 12:
            # Extract the ROI of the potential match
            roi = gray_frame[best_match['y']:best_match['y']+best_match['h'], 
                             best_match['x']:best_match['x']+best_match['w']]
            
            if roi.size == 0: return None
            
            # A. Template Rejection
            kp_roi, des_roi = WatermarkDetector._orb.detectAndCompute(roi, None)
            if des_roi is not None: 
                for (kp_neg, des_neg, _) in WatermarkDetector._negative_features:
                    if des_neg is None: continue
                    neg_matches = WatermarkDetector._bf.match(des_neg, des_roi)
                    neg_good = [m for m in neg_matches if m.distance < 50]
                    
                    if len(neg_good) > 8: 
                        logger.warning(f"🛑 Match rejected: Resembles known FALSE POSITIVE (Matches: {len(neg_good)})")
                        return None

            # B. ML Model Validation (if available)
            if WatermarkDetector._model:
                features = WatermarkLearner.extract_features(frame, best_match, {"confidence": best_match['confidence'], "matches": max_good_matches})
                if features:
                    # Convert to DataFrame for prediction
                    try:
                        import pandas as pd
                        X_pred = pd.DataFrame([features])
                        prediction = WatermarkDetector._model.predict(X_pred)[0]
                        
                        if prediction == 0:
                            logger.warning(f"🛑 Match rejected by ML Model (Predicted False Positive)")
                            return None
                        else:
                            logger.info(f"✅ Match verified by ML Model")
                    except: pass

            logger.info(f"👁️ ORB detected learned watermark (Matches: {max_good_matches})")
            return best_match
            
        return None



# Helper functions
def learn(frame, coords, is_positive=True):
    WatermarkLearner.save_template(frame, coords, is_positive)

def detect(frame):
    return WatermarkDetector.detect(frame)

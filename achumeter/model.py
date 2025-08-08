import numpy as np
import os, joblib

THIS_DIR = os.path.dirname(__file__)

class AchuModel:
    def __init__(self):
        clf_path = os.path.join(THIS_DIR, 'models', 'intensity_clf.joblib')
        reg_path = os.path.join(THIS_DIR, 'models', 'time_reg.joblib')
        if os.path.exists(clf_path) and os.path.exists(reg_path):
            try:
                self.clf = joblib.load(clf_path)
                self.reg = joblib.load(reg_path)
                self.mode = 'trained'
            except Exception:
                self.clf = None
                self.reg = None
                self.mode = 'rule'
        else:
            self.clf = None
            self.reg = None
            self.mode = 'rule'

    def predict(self, features):
        if self.mode == 'trained' and self.clf is not None and self.reg is not None:
            intensity = self.clf.predict([features])[0]
            time_pred = float(self.reg.predict([features])[0])
            return max(0.0, time_pred), intensity
        else:
            return self.rule_predict(features)

    def rule_predict(self, features):
        rms = float(features[13])
        centroid = float(features[16])
        if rms > 0.08 or centroid > 3000:
            intensity = 'high'
        elif rms > 0.03:
            intensity = 'medium'
        else:
            intensity = 'low'
        time_pred = max(0.1, 5.0 - (rms * 40.0))
        time_pred = min(time_pred, 10.0)
        return time_pred, intensity

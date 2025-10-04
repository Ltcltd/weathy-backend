"""
Extreme Event Detection (optional)
Can detect anomalies in predictions
"""

class ExtremeDetector:
    def __init__(self):
        self.thresholds = {
            'rain': 0.8,
            'hot': 0.7,
            'cold': 0.7,
            'windy': 0.75
        }
    
    def detect(self, probability, variable):
        """Detect if probability indicates extreme event"""
        threshold = self.thresholds.get(variable, 0.75)
        return probability > threshold
    
    def get_severity(self, probability, variable):
        """Get severity level"""
        if probability > 0.9:
            return "severe"
        elif probability > 0.8:
            return "high"
        elif probability > 0.7:
            return "moderate"
        else:
            return "low"

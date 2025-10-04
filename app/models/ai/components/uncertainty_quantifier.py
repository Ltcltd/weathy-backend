import numpy as np

class UncertaintyQuantifier:
    def calculate(self, predictions_dict_or_list, lead_time_months):
        """Calculate uncertainty based on model disagreement and lead time
        
        Args:
            predictions_dict_or_list: dict of model predictions or list of values
            lead_time_months: forecast lead time in months
        """
        # Handle both dict and list inputs
        if isinstance(predictions_dict_or_list, dict):
            predictions_list = list(predictions_dict_or_list.values())
        else:
            predictions_list = predictions_dict_or_list
        
        # Ensemble variance (model disagreement)
        variance = np.var(predictions_list)
        
        # Lead time adjustment
        if lead_time_months < 6:
            time_factor = 1.0
        elif lead_time_months < 12:
            time_factor = 1.5
        else:
            time_factor = 2.5
        
        uncertainty = np.sqrt(variance) * time_factor
        return float(np.clip(uncertainty, 0.05, 0.4))
    
    def get_confidence(self, uncertainty):
        """Map uncertainty to confidence level"""
        if uncertainty < 0.1:
            return "high"
        elif uncertainty < 0.2:
            return "medium"
        else:
            return "low"

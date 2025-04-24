import numpy as np
import logging

logger = logging.getLogger(__name__)

def safe_predict(model, X_input, default_value=None):
    if model is None:
        logger.warning("Model is None, cannot make prediction")
        return default_value
    
    try:
        if X_input is None or not isinstance(X_input, np.ndarray) or X_input.size == 0:
            logger.warning(f"Invalid X_input for prediction: {type(X_input)}, size: {getattr(X_input, 'size', 'unknown')}")
            return default_value
        
        X_modified = X_input.copy()
        
        expected_features = None
        if hasattr(model, 'n_features_in_'):
            expected_features = model.n_features_in_
        elif hasattr(model, 'model') and hasattr(model.model, 'n_features_in_'):
            expected_features = model.model.n_features_in_         
        if expected_features is not None:
            input_features = X_modified.shape[1]           
            if input_features > expected_features:
                logger.warning(f"Input has {input_features} features, but model expects {expected_features}. Truncating.")
                X_modified = X_modified[:, :expected_features]
            elif input_features < expected_features:
                logger.warning(f"Input has {input_features} features, but model expects {expected_features}. Padding.")
                padding = np.zeros((X_modified.shape[0], expected_features - input_features))
                X_modified = np.hstack((X_modified, padding))    
        predictions = model.predict(X_modified)
        if predictions is None or not isinstance(predictions, np.ndarray) or predictions.size == 0:
            logger.warning(f"Empty prediction result: {type(predictions)}, size: {getattr(predictions, 'size', 'unknown')}")
            return default_value
        return predictions[0]
        
    except Exception as e:
        logger.error(f"Error in safe_predict: {str(e)}")
        return default_value
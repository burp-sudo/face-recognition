import face_recognition_models
import os

# Debug: show model paths
print("Model path:")
print("  Face recognition model:", face_recognition_models.face_recognition_model_location())
print("  Landmark predictor:", face_recognition_models.pose_predictor_model_location())

# Check that files exist
assert os.path.exists(face_recognition_models.face_recognition_model_location()), "Face recognition model not found!"
assert os.path.exists(face_recognition_models.pose_predictor_model_location()), "Landmark model not found!"

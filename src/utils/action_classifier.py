import sys
sys.path.append('.')
import torch
import joblib

l2gen = joblib.load('label_mapping.pt')


def evaluate_gait_classifier(model, dataset, iterator, parameters):
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            for key in batch.keys():
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(model.device)
            prediction = model.predict(batch)
            threshold = 0.5  # You may adjust this threshold as needed
            binary_predictions = (prediction > threshold).int()
            # True positives, false positives, and false negatives
            TP = torch.sum((binary_predictions == 1) & (batch['labels'] == 1), dim=0)
            FP = torch.sum((binary_predictions == 1) & (batch['labels'] == 0), dim=0)
            FN = torch.sum((binary_predictions == 0) & (batch['labels'] == 1), dim=0)
            # Precision and recall
            precision = TP / (TP + FP + 1e-9)  # Adding a small value to avoid division by zero
            recall = TP / (TP + FN + 1e-9)

            # F1 score
            f1_score = 2 * (precision * recall) / (precision + recall + 1e-9)
            return f1_score
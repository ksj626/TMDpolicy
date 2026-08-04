# Evaluation

`metrics.py` separates step-integrated average precision from trapezoidal PR-AUC and provides 95% Wilson intervals. `outcomes.py` keeps task success, environment termination/truncation, and local limits distinct. It has no training gradients or writes.


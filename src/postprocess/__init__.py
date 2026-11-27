"""Hậu xử lý score cho M1/M2 DiffMul: calibration, SWA và ensemble.

Các module trong package này chỉ làm việc với logits/probabilities đã tính sẵn,
không can thiệp vào protocol huấn luyện gốc. Mọi ngưỡng đều được chốt trên
validation (hoặc OOF trên validation), không bao giờ tune trên test.
"""

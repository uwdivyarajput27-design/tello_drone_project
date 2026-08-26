"""
reid_extractor.py
OSNet-based person re-identification embedding extractor (torchreid).
"""

import torch
from torchreid.utils import FeatureExtractor

extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path='',      # empty string -> auto-download pretrained weights
    device='cuda' if torch.cuda.is_available() else 'cpu'
)


def extract_embedding(person_crop):
    """
    person_crop: BGR image (OpenCV) cropped to a single person.
    returns: 1D numpy embedding.
    """
    embeddings = extractor(person_crop)
    return embeddings[0].cpu().numpy()

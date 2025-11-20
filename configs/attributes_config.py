import torch
from configs.paths_config import edit_paths


edit_attributes = {
    # Wu et al., "Stylespace Analysis: Disentangled Controls For StyleGAN Image Generation", CVPR, 2021.
    # https://arxiv.org/pdf/2011.12799.pdf  
    # https://github.com/xrenaa/StyleSpace-pytorch  
    "hair": {
        "level": "style",
        "index": (6, 364)
    },       
    "beard_goatee": {
        "level": "style",
        "index": (9, 421)
    },      
    "mouth_smiling": {
        "level": "style",
        "index": (6, 501)
    },
    "mouth_lipstick": {
        "level": "style",
        "index": (15, 45)
    },
    "eyes_makeup": {
        "level": "style",
        "index": (12, 414)
    },      
    "gaze": {
        "level": "style",
        "index": (9, 409)
    },
    "eyebrows": {
        "level": "style",
        "index": (8, 28)
    },    
    "gender": {
        "level": "style",
        "index": (9, 6)
    },   
    # Shen et al., "Interpreting the Latent Space of GANs for Semantic Face Editing", CVPR, 2020.
    # https://arxiv.org/pdf/1907.10786.pdf
    # https://github.com/genforce/interfacegan
    "pose": {
        "level": "latent",
        "direction": torch.load(edit_paths['pose'], map_location='cpu')
    },
    "age": {
        "level": "latent",
        "direction": torch.load(edit_paths['age'], map_location='cpu')
    },
}
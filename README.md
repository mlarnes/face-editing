# Face Editing and Morphing

<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" height=22.5></a>

A powerful deep learning framework for semantic face editing and smooth face morphing using StyleGAN2 and HyperStyle. This project enables high-quality manipulation of facial attributes and seamless interpolation between multiple faces in the latent space.

## Examples

### Face Editing
![Face Editing Example](./docs/editing_256x256.gif)

### Face Morphing
![Face Morphing Example](./docs/morphing_256x256.gif)

## Features

### 🎨 Face Editing
Semantic editing of facial attributes with fine-grained control:
- **Age manipulation** - Make faces appear younger or older
- **Facial expressions** - Adjust smile intensity and pose
- **Cosmetic attributes** - Modify hair, beard, lipstick, and eye makeup
- **Gender characteristics** - Subtle gender adjustments
- **Gaze direction** - Control eye gaze and eyebrow position

### 🔄 Face Morphing
Create smooth transitions between multiple faces:
- **Latent space interpolation** - Seamless morphing using StyleGAN2's latent space
- **Multi-face sequences** - Generate morphing videos between multiple faces
- **High-quality output** - Preserves facial identity and structure

## Quick Start

### Try it on Google Colab

- **Face Editing**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlarnes/face-editing/blob/main/notebooks/editing.ipynb)
- **Face Morphing**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlarnes/face-editing/blob/main/notebooks/morphing.ipynb)

### Local Installation

#### Prerequisites
- Python 3.6+
- CUDA-capable GPU (recommended)
- Conda or pip

#### Setup

1. **Clone the repository**
```bash
git clone https://github.com/mlarnes/face-editing.git
cd face-editing
```

2. **Create conda environment**
```bash
conda env create -f environment/environment.yaml
conda activate face-editing
```

3. **Download pretrained models**
Download the pretrained models archive and extract it to the project root:
```bash
# Models will be placed in pretrained_models/ directory
wget "https://nextcloud.univ-lille.fr/index.php/s/8RrH7MzLGFfwMjK/download/pretrained_models.tar.xz"
tar -xvJf pretrained_models.tar.xz
rm pretrained_models.tar.xz
```

4. **Install additional dependencies**
```bash
pip install ninja
```

## Usage

### Face Editing

```python
from models.face_model import FaceModel
from utils.common import *
from utils.data_utils import *
from utils.inference_utils import *
from utils.editing_utils import *
from configs.attributes_config import edit_attributes
from PIL import Image

# Initialize model
face_model = FaceModel(device="cuda")

# Load image
image = Image.open("path/to/face.jpg")

# Define edits
edits = edit_attributes.copy()
edits["age"]["factor"] = 2.5  # Make older
edits["smile"]["delta"] = 5.0  # Increase smile
edits["hair"]["delta"] = 10.0  # Modify hair

# Run editing
data = run_prediction(face_model, [image], return_weights_deltas=True)
latent = data["latent"][0]
weights_deltas = data["weights_deltas"][0]

# Apply edits
for name, config in edits.items():
    if config["level"] == "latent" and config.get("factor"):
        latent = edit_latent(latent, config["direction"], config["factor"])
    elif config["level"] == "style" and config.get("delta"):
        if style is None:
            style = face_model.latent2style(latent)
        style = edit_style(style, config["index"], config["delta"])

# Generate edited image
edited = face_model.decoder(latent=latent, weights_deltas=weights_deltas, style=style, resize=False)
result = tensor2im(edited[0])
```

### Face Morphing

```python
from models.face_model import FaceModel
from utils.morphing_utils import interpolation
from utils.data_utils import create_video
from PIL import Image

# Initialize model
face_model = FaceModel(device="cuda")

# Load images
images = [Image.open(f"face_{i}.jpg") for i in range(5)]

# Encode faces
data = run_prediction(face_model, images, return_weights_deltas=True)

# Create morphing sequence
n_frames = 30
morphed_images = []

for i in range(len(images) - 1):
    p_A = {"latent": data["latent"][i], "weights_deltas": data["weights_deltas"][i]}
    p_B = {"latent": data["latent"][i+1], "weights_deltas": data["weights_deltas"][i+1]}
    
    pts = interpolation(p_A, p_B, n_frames)
    
    for p in pts:
        x = face_model.decoder(latent=p["latent"], weights_deltas=p["weights_deltas"], resize=False)
        morphed_images.append(tensor2im(x[0]))

# Save video
create_video(morphed_images, "morphing_output.mp4", mode="cv2")
```

## Available Attributes

The framework supports editing of the following attributes:

### Latent Space Edits (InterfaceGAN)
- **Age** - Control age appearance
- **Pose** - Adjust head pose and orientation
- **Smile** - Modify smile intensity

### Style Space Edits (StyleSpace)
- **Hair** - Change hair style and appearance
- **Beard/Goatee** - Add or remove facial hair
- **Mouth (Smiling)** - Control smile expression
- **Mouth (Lipstick)** - Add or modify lipstick
- **Eyes (Makeup)** - Apply eye makeup
- **Gaze** - Control eye gaze direction
- **Eyebrows** - Adjust eyebrow shape and position
- **Gender** - Subtle gender characteristics

## Architecture

This project builds upon several state-of-the-art techniques:

- **HyperStyle**: For high-fidelity face inversion using hypernetworks
- **StyleGAN2**: As the generator backbone for face synthesis
- **InterfaceGAN**: For semantic editing in the latent space
- **StyleSpace**: For fine-grained control in the style space

### Key Components

- `FaceModel`: Main model class wrapping HyperStyle encoder and StyleGAN2 decoder
- `editing_utils`: Functions for latent and style space manipulation
- `morphing_utils`: Interpolation functions for smooth transitions
- `inference_utils`: Utilities for running predictions and inversions

## Project Structure

```
face-editing/
├── configs/              # Configuration files
│   ├── attributes_config.py  # Attribute editing configurations
│   └── paths_config.py      # Model paths configuration
├── criteria/             # Loss functions
│   ├── lpips/           # LPIPS perceptual loss
│   ├── id_loss.py       # Identity preservation loss
│   └── ...
├── models/               # Model implementations
│   ├── face_model.py    # Main FaceModel class
│   ├── hyperstyle.py    # HyperStyle architecture
│   ├── encoders/        # Encoder implementations
│   ├── stylegan2/       # StyleGAN2 generator
│   └── ...
├── notebooks/           # Jupyter notebooks
│   ├── editing.ipynb    # Face editing notebook
│   └── morphing.ipynb   # Face morphing notebook
├── utils/               # Utility functions
│   ├── editing_utils.py # Editing operations
│   ├── morphing_utils.py # Morphing operations
│   ├── inference_utils.py # Inference pipeline
│   └── ...
└── environment/         # Conda environment file
```

## Requirements

Key dependencies:
- PyTorch 1.10.0+
- torchvision 0.11.1+
- dlib 19.22.1+
- opencv-python 4.2.0+
- numpy, scipy, PIL
- See `environment/environment.yaml` for complete list

## Technical Details

### Encoding Process
1. **Face Alignment**: Automatic face detection and alignment using MTCNN
2. **Inversion**: HyperStyle encoder maps image to StyleGAN2 latent space
3. **Fine Encoding** (optional): Additional optimization for better reconstruction

### Editing Process
- **Latent edits**: Direct manipulation in W+ latent space using InterfaceGAN directions
- **Style edits**: Fine-grained control in StyleSpace for specific attributes

### Morphing Process
- Interpolation in latent space between multiple encoded faces
- Smooth transitions preserving identity and structure
- Support for multi-face sequences with looping

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project is built upon the following excellent works:

- **[HyperStyle](https://github.com/yuval-alaluf/hyperstyle)** by Yuval Alaluf et al. - High-fidelity face inversion
- **StyleGAN2** by Karras et al. - Generative adversarial network architecture
- **InterfaceGAN** by Shen et al. - Semantic face editing in latent space
- **StyleSpace** by Wu et al. - Disentangled controls for StyleGAN

## Citation

If you use this code in your research, please cite the original HyperStyle paper:

```bibtex
@inproceedings{hyperstyle2021,
  title={HyperStyle: StyleGAN Inversion with HyperNetworks for Real Image Editing},
  author={Alaluf, Yuval and Tov, Or and Mokady, Ron and Gal, Rinon and Bermano, Amit H},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```

## Contact

For questions, suggestions, or collaborations, please contact:
- **Email**: estephe.arnaud@univ-lille.fr

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

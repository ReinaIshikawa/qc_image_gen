
# Installation

```bash
pip install torch torchvision matplotlib pillow
```

# How to Load the Dataset

Refer to the example code in `example_load_data.py` for details on how to load and use the dataset.
The loaded dataset provides pairs of:
- Digit image (resized)
  - Default size: 16×16
  - You can change the image size in config.yaml
  - Image is halftoned — pixel values are either 0 or 1
- Label
  - Integer from 0 to 9
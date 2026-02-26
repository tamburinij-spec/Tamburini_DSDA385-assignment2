# Tamburini DSDA385 Assignment 2

This repository contains the code and data organization for the second assignment
in the DSDA385 course. Currently the focus is on training a pedestrian detection
model using the PennFudanPed dataset (via a Faster R-CNN implementation). The
project has been refactored to follow a modular layout so that additional
models/datasets can be added later.

## Updated Structure

```
Tamburini_DSDA385-assignment2/
│
├── README.md
├── requirements.txt
├── config/
│   ├── faster_rcnn.yaml
│   ├── yolo.yaml
│   └── dataset.yaml
│
├── data/
│   ├── pennfudan/
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── masks/
│   │   ├── val/...
│   │   └── test/...
│   └── pets_subset/        # placeholder for later experiments
│
├── src/
│   ├── datasets/
│   │   ├── pennfudan.py
│   │   └── pets.py
│   │
│   ├── models/
│   │   ├── faster_rcnn.py
│   │   └── yolo_wrapper.py
│   │
│   ├── engine/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── metrics.py
│   │
│   ├── utils/
│   │   ├── transforms.py
│   │   ├── visualization.py
│   │   └── device.py
│   │
│   └── main.py
│
├── experiments/
│   ├── faster_rcnn_pennfudan/
│   └── yolo_pets/
│
├── outputs/
│   ├── checkpoints/
│   ├── predictions/
│   └── logs/
│
└── report/
    └── assignment2_report.pdf
```

## Setup Instructions

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare data**
   - PennFudan dataset should be placed under `data/pennfudan`. Use the
     `src/utils/organize_dataset.py` script to split and copy images if needed.

3. **Run training**
   ```bash
   python src/main.py
   ```

## Notes

- Configuration files are YAML and live under `config/`.
- This repo currently supports a Faster R-CNN segmentation/detection model
  trained on pedestrians; the structure is extensible to other models/datasets.

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd pytorch-project
   ```

2. **Create a virtual environment:**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the application, execute the following command:

```
python src/main.py
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
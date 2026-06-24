<p align="center">
  <img alt="Delirium Tremens" src="https://github.com/Amloii/Delirium-Tremens/blob/main/data/descarga.jfif" width="120" style="border-radius: 16px;">
  <h1 align="center">Delirium Tremens</h1>
  <p align="center"><b>Zero-shot dataset generator for object detection.</b></p>
  <p align="center">Create COCO-format datasets from raw images using OWL-ViT zero-shot object detection.</p>
</p>

<p align="center">
  <a href="https://github.com/Amloii/Delirium-Tremens"><img alt="Stack" src="https://img.shields.io/badge/stack-Python%20%7C%20HuggingFace%20%7C%20OWL--ViT-25601B?style=flat-square&labelColor=ffffff&color=25601B"></a>
  <a href="https://github.com/Amloii/Delirium-Tremens"><img alt="Created" src="https://img.shields.io/badge/created-December%202022-000000?style=flat-square&labelColor=ffffff&color=000000"></a>
  <a href="https://github.com/Amloii/Delirium-Tremens"><img alt="Status" src="https://img.shields.io/badge/status-discontinued-ef4444?style=flat-square&labelColor=ffffff&color=ef4444"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-blue?style=flat-square&labelColor=ffffff&color=blue"></a>
</p>

<p align="center">
  <a href="#-about">About</a> •
  <a href="#-how-it-works">How It Works</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-project-structure">Structure</a>
</p>

<br>

> **Created December 2022 as a pet project during the Adsviu era within Grupo NextChance.**  
> This project has been **discontinued**. Modern approaches — such as foundation models, synthetic data generation with LLMs, and fine-tuned vision transformers — offer more practical dataset generation workflows. The code remains available for reference.

---

## 🧐 About

Delirium Tremens is a tool for rapidly bootstrapping object detection datasets without manual annotation. It uses zero-shot object detection via [OWL-ViT](https://huggingface.co/google/owlvit-base-patch32) (HuggingFace) to automatically detect objects in raw images and export the results in COCO JSON format.

The tool was born from a practical need during the early stages of Adsviu: labeling training data for a visual product recognition system. Manually annotating thousands of images was infeasible, so this pipeline automated the process using the best zero-shot model available at the time.

The repo contains two iterations: an initial version (`src/`) and a refactored version (`new_src/`) with dataset merging capabilities.

---

## ⚙️ How It Works

1. **Zero-shot detection** — OWL-ViT scans raw images and detects objects by textual category prompts (e.g. "a image of a bottle").
2. **NMS filtering** — Non-maximum suppression removes overlapping detections and filters by confidence + IoU thresholds.
3. **COCO transformation** — Detections are converted to COCO JSON with train/test splits.
4. **Dataset merging** (new_src) — New detections can be appended to an existing COCO dataset, with automatic annotation indexing and duplicate image removal.

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/Amloii/Delirium-Tremens.git
cd Delirium-Tremens

# Install dependencies
pip install torch torchvision transformers pandas pillow scikit-learn tqdm
```

### Usage (original — `src/`)

```python
from ZeroShotObjectDetection import ZeroShotObjectDetection
from CocoTransformer import CocoTransformer

config = {
    'raw_image_folder': './raw_images/',
    'model_zeroshotobjectdetection': 'google/owlvit-base-patch32',
    'categories_list': ['bottle'],
    'conf_threshold': 0.2,
    'iou_threshold': 0.2,
    'test_split': 0.2,
    'image_folder': './output/',
}

detector = ZeroShotObjectDetection(config)
transformer = CocoTransformer(config)

detections = detector.predict(images_dir=config['raw_image_folder'])
transformer.transform(detections)
```

### Usage (refactored — `new_src/`)

```python
from ZeroShotObjectDetection import ZeroShotObjectDetection
from Preprocess import Preprocess
from CocoDatasetMerger import CocoDatasetMerger

config = {
    'new_images_folder': './new_images/',
    'dataset_input_folder': './existing_dataset/',
    'new_category': 'Wine',
    'include_previous_images': False,
    'model_zeroshotobjectdetection': 'google/owlvit-base-patch32',
    'categories_list': ['bottle'],
    'conf_threshold': 0.5,
    'iou_threshold': 0.2,
}

object_detector = ZeroShotObjectDetection(config)
coco_merger = CocoDatasetMerger(config)

Preprocess.remove_duplicated_images(config['new_images_folder'])
# ... iterate images, detect, merge
coco_merger.save_dataset(overwrite=True)
```

---

## 📁 Project Structure

```
├── src/
│   ├── main.py                       # Original pipeline entry point
│   ├── ZeroShotObjectDetection.py    # OWL-ViT detector + NMS
│   └── CocoTransformer.py            # Detection → COCO JSON
├── new_src/
│   ├── main.py                       # Refactored entry point
│   ├── ZeroShotObjectDetection.py    # Improved detector class
│   ├── CocoDatasetMerger.py          # Merge new detections into existing datasets
│   └── Preprocess.py                 # Duplicate removal
├── data/
│   └── descarga.jfif                 # Logo
└── README.md
```

---

## 📝 License

MIT — see [LICENSE](LICENSE).

---

## 👤 Author

**Daniel Gómez Domínguez** — AI Systems Architect & Director of AI.

Developed as a side project during his tenure as Technical Founder of Adsviu (Grupo NextChance), where the need for rapid dataset creation for visual product recognition drove the experiment.

[GitHub](https://github.com/Amloii) · [LinkedIn](https://linkedin.com/in/danigdominguez) · [Portfolio](https://amloii.github.io)

<br>

---

<p align="center">
  <sub>Delirium Tremens · 2022–2023 — Discontinued, but the name lives on.</sub>
</p>

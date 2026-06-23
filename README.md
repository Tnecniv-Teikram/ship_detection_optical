# 🚢 High-Resolution Optical Ship Detection with Deep Learning
### A practical tutorial for GeoAI analysts working with satellite imagery

This repository contains ship_detection.ipynb, a hands-on notebook for detecting ships in **high-resolution optical remote sensing imagery** using a deep learning object detection workflow. The tutorial is designed for **GeoAI analysts** who want to better understand how deep learning can be applied to satellite data for maritime monitoring and remote sensing analysis.

---

## 📌 Overview

Ship detection in optical satellite imagery is an important GeoAI application with use cases in:

- Maritime surveillance  
- Port and coastal activity monitoring  
- Marine traffic analysis  
- Illegal fishing detection  
- Environmental monitoring  

This tutorial focuses on **high-resolution optical imagery**, where ships are identified from their visual appearance, shape, and surrounding context. In optical remote sensing, ship detection can be challenging because of **complex marine backgrounds, large variation in ship size, dense scenes, and visually cluttered coastlines**.  

---

## 🧠 What this tutorial demonstrates

This notebook follows a **deep learning object detection workflow** for optical ship detection. Based on the accessible implementation details, the workflow includes:

1. **Preparing annotated ship data**
   - Reading image annotations in XML/VOC format
   - Extracting bounding boxes and image metadata
   - Converting annotations into the training format required by the detector

2. **Creating a training-ready dataset**
   - Organizing image paths for training and validation
   - Generating label files for object detection

3. **Configuring the object detection model**
   - Setting class definitions
   - Preparing model configuration files
   - Adapting the detector to a ship detection task

4. **Training the detector**
   - Running deep learning training on the prepared dataset
   - Monitoring losses and detection metrics

5. **Evaluating and visualizing predictions**
   - Reviewing precision, recall, and mAP
   - Comparing predictions with ground-truth annotations

---

## ⚙️ Model and implementation

The accessible tutorial content associated with this workflow uses the **YOLOv7** object detection framework and trains a **two-class detector** for:

- `dock`
- `ship`

The implementation also shows Python-based preprocessing for converting **VOC/XML annotations** into the label format expected by the detector.

### Important note on the software stack
This workflow is implemented primarily as a **deep learning / computer vision pipeline**.  
The accessible implementation details explicitly show:

- Python utilities such as `os`
- XML parsing with `xml.etree.ElementTree`
- Configuration of the **YOLOv7** repository and training scripts

**No separate dedicated GeoAI Python library is explicitly shown in the accessible implementation details.**  
Instead, this tutorial teaches a practical pattern that GeoAI analysts can apply to remote sensing data using a general deep learning object detection framework.

---

## 🛰️ Dataset context

The referenced implementation uses **ShipRSImageNet**, described as a large-scale fine-grained dataset for ship detection in **high-resolution optical remote sensing imagery**.

Key points from the accessible tutorial description:

- Dataset: **ShipRSImageNet**
- Approximate size: **3,435 images**
- Annotated ship instances: **17,573**
- Source imagery includes multiple sensors and platforms
- For simplicity, the workflow reduces the problem to **two classes**: `ship` and `dock`

This makes the tutorial especially useful for analysts who want to understand how to go from **annotated satellite imagery** to a trained detector.

---
## Example results

<img width="1811" height="960" alt="image" src="https://github.com/user-attachments/assets/f4f72db3-94b1-4729-9b20-5a2a7d0a8eb2" />

---



## 🧭 End-to-end workflow

```mermaid
flowchart LR
    A[High-resolution optical imagery] --> B[Annotation parsing]
    B --> C[Convert VOC/XML to detection labels]
    C --> D[Prepare train/validation files]
    D --> E[Configure model]
    E --> F[Train detector]
    F --> G[Evaluate results]
    G --> H[Visualize predictions]

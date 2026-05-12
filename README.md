# Pipeline de Vision Artificielle — Classification Médicale

## Description
Pipeline complet de vision artificielle appliqué à la classification 
de radiographies thoraciques (Normal vs Pneumonie).

**Dataset :** Kaggle Chest X-Ray Images (Pneumonia)  
**Module :** Vision Artificielle  

## Pipeline
Image → Prétraitement → Segmentation → Features → Classification → Évaluation

## Résultats
| Modèle | Accuracy | F1-Score |
|--------|----------|----------|
| KNN (k=5) | 86.2% | 86.1% |
| SVM (RBF) | 85.0% | 85.0% |
| MLP | 83.8% | 83.3% |

## Installation
```bash
pip install numpy pandas matplotlib opencv-python scikit-image scikit-learn scipy
```

## Utilisation
Ouvrir `projet_vision_medicale.ipynb` dans VS Code ou Jupyter

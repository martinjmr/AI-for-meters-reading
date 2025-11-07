# AI-for-meters-reading-

## Introduction
C’est ce moment de l’année où vous devez prendre rendez-vous pour qu’un agent vienne relever votre compteur d’eau.  
Vous êtes débordé, souvent absent, et c’est une vraie galère.  

Et si une simple photo suffisait ?  
Vous la prenez, l’envoyez, et un algorithme lit automatiquement les chiffres.  
Plus besoin de rendez-vous, plus de contraintes.  

C’est exactement ce que nous avons développé cette semaine 

Nous avons travaillé sur **795 images de compteurs d’eau**.  
Ces photos sont parfois floues, retournées, peu lisibles, voire prises de loin.  
L’objectif : **lire automatiquement les trois chiffres visibles devant le cadran rouge**, correspondant à la consommation en m³ du particulier.

---

## Dataset
- 795 images de compteurs d’eau
- Certaines images très bruitées ou mal orientées
- Jeu de données **très réduit** pour une tâche de **vision par ordinateur**

---

## Preprocessing

### 1. **Découpage des cadrans**
Nous avons utilisé **Roboflow** pour annoter manuellement **150 images**, puis entraîné un premier modèle **YOLOv8** afin de détecter les cadrans de chiffres.

Ce premier modèle a généré **320 images bien découpées**, puis nous avons **réentraîné** un second modèle YOLO sur ce nouveau jeu.  
Résultat final : **726 images sur 795 correctement recadrées.**

#### Paramètres YOLO :
- `rect=False` (images presque carrées)
- Optimizer : `AdamW(lr=0.002, momentum=0.9)`
- Loss : `box + cls + dfl`
- **AdamW** = version d’Adam avec décroissance réelle des poids *(weight decay)* pour une meilleure généralisation.

---

### 2. **Alignement des cadrans**
Nous avons voulu mettre toutes les images à l’horizontale pour faciliter la lecture automatique.

Méthode :
- Deux labels créés sur Roboflow : une **boîte rouge** et une **boîte noire**
- Détection de ces zones par YOLOv8
- Récupération des coordonnées et calcul des **centres**
- Calcul de l’angle entre les deux centres, puis **rotation automatique** pour réaligner chaque cadran

Ce processus a permis de rendre les chiffres homogènes dans le dataset.

---

### 3. **Suppression de la partie rouge**
Une étape de post-traitement a été ajoutée pour supprimer la partie rouge du compteur (zone non informative pour la prédiction).

---

## Modeling

### CNN personnalisé
Nous avons construit un petit **réseau de neurones convolutionnel (CNN)** :
- Trois blocs convolutionnels  
- Batch normalization + ReLU + MaxPooling  
- Trois têtes indépendantes  
- `CrossEntropyLoss`  
- Dropout 30%  

**Résultat :**
- Accuracy exacte (3 chiffres corrects) : **0.0068**

### Limites rencontrées avec ResNet
Nous avons testé **ResNet**, mais le faible nombre d’images et la complexité du modèle ont entraîné un sur-apprentissage sans amélioration notable.

### Solution retenue : **EasyOCR**
Finalement, nous avons intégré **EasyOCR**, un modèle pré-entraîné de reconnaissance de texte, qui s’est avéré **le plus efficace** pour lire directement les chiffres sur les cadrans.

---

## Résultats
- YOLOv8 : détection et alignement automatisé des cadrans
- EasyOCR : lecture des chiffres sur images alignées
- Pipeline complet permettant d’obtenir le nombre de m³ à partir d’une simple photo

---

## Difficultés rencontrées
- Faible quantité de données (795 images seulement)
- Variations d’angle, de luminosité et de netteté
- Rotation automatique complexe à généraliser

---

## Pistes d’amélioration
- Découper **chaque chiffre individuellement**
- Améliorer le **prétraitement d’image** (rehaussement du contraste, suppression du bruit)
- Enrichir le dataset avec des **augmentations artificielles**
- Fine-tuning du modèle OCR sur des chiffres de compteurs

---

## Installez les indépendences 
pip install -r requirements.txt


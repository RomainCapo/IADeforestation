# IADeforestation
## Enumération des notebooks
* **ImageProccesing.ipynb :** 
    * Découpage des images Sentinel2
    * Attribution des labels au images
    * Enregistrement des images sur le disque 
    * Calcul de la moyenne et écart-types des images
* **DatasetVisualisation.ipynb :**
    * Visualisation des images de différentes classes
* **Statistics.ipynb :**
    * Calcul de statistiques sur la répartition des classes
    * Calcul de statistiques sur les bandes des images
* **SpatialCrossVal.ipynb :**
    * Affichage des labels sur la carte du Vietnam
    * Séparation du jeu d’entrainement, validation et test avec une séparation spatiale
* **TrainingTensorflowHub.ipynb :**
    * Entrainement des modèles TensorFlow Hub Resnet50 3 bandes pour café contre reste et culture contre non-culture
* **TrainingTransferMulti.ipynb :**
    * Entrainement des modèles de transfer learning DenseNet201 13 bandes pour café contre reste et culture contre non-culture
* **TrainingBandsVariation.ipynb :**
    * Entrainement des modèles de transfer learning DenseNet201 en faisant varier le nombre de bande pour café contre reste
* **TrainingFromScratchMulti.ipynb :**
    * Entrainement du modèle DenseNet201 13 bandes sans poids pré entrainé pour café contre reste et culture contre non-culture
* **GradCam.ipynb :**
    * Affichage GradCam pour les différentes couches de convolution du modèle de transfer learning DenseNet201
* **ModelsComparison.ipynb :** 
    * Comparaison des scores des différentes expériences réalisées dans ce projet
    
## Enumération autres fichiers
* **admin :**
    * Rapport
    * Journeaux de travails
* **datasets :**
    * Datsets au format CSV utilisé pour les différentes expériences
* **district :**
    * Contient les différents fichiers permettant d'afficher la carte du Vietnam avec GeoPandas
* **IAdeforestation :**
    * Contient différents fichiers avec du code en commun entre les différents notebook 
    * preprocessing.py : Méthodes utilitaires pour le découpage des image, la labélisation, l'enregistrement sur le disque et la normalisation
    * spatial_cross_val.py : Méthodes utilitaires pour le découpage du datasets de manière spatiale et méthode de visualisation des points sur la carte
    * tools.py : Méthode utilitaires pour affichage des images
    * training.py : Contient moyenne et écart-type Eurosat et Vietnam, contient différents version de génrateur utilisé, contient métrique custom
    * grad_cam.py : Méthode utilitaires pour calcul et visualisation GardCam.
* **labels :**
    * Contient les labels au format ShapeFile fournit par le CIAT.
* **notebook_images :**
    * Contient les images présent dans les différents notebooks.
* **saved_models :**
    * Contient l'export des modèles (format h5), de l'historique d'entrainement (format .npy) et les scores (format .json)
* **SentinelImages :**
    * Contient les images brutes Sentinel2 de taille 10980x10980
* **spring_images :**
    * Contient les images découpées de taille 64x64
* **spring_images_32 :**
    * Contient les images découpées de taille 32x32
* **transfer_learning_model :**
    * Contient les différents modèle de transfer learning provenant du travail de E.Ransome au format hdf5 
    * script : 
        * Contient les notebooks utilisé dans le travail de E.Ransome
* **map.geojson** : Fichier GEOJSON contenant un MultiPolygon qui englobe 5 images Sentinel2 contenant tous les labels fournit par le CIAT.
* **download_sentinel.py :** Script permettant de telécharger les images Sentinel2 brut à partir du fichier map.geojson
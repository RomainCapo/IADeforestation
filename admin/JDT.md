# JDT
## Semaine 1 - 22.02.2021 - 26.02.2021 [10h00]
* **Lecture ancien travail de bachelor portant sur la meme thematique - 3h00**
    * Identification des objectifs, inputs, outputs, problème rencontré, subilité
    * Reproduction de certain example du rapport
* **Lecture ancien travail de master d'ou sera repris le modèle pour le transfer learning - 1h00**
    * Identification des objectifs du travail, inputs, outputs
* **Prise en main des images Sentinel-2 et de la librairie sentinelsat - 3h00**
    * Création d'un compte sur copernicus
    * Installation sentinelsat et rasterio
    * Réalisation d'un script de test pour télécharger et afficher des images de Sentinel-2 (basé sur le rapport du travail de bachelor)
    * Manipulation basiques des images avec rasterio (selection bande, localisation, affichage, ...)
* **Documentation à propos des bands et du type de produits (2a et 1c) - 2h30**
    * Telechargement de SNAP pour réaliser des tests sur les images sentinels
    * Lecture de documentation sur la différence entre le type Sentinel 2a et 1c
    * Lecture documentation sur le nombre de bands, à priori, comme le modèle de transfer learning du TM doit être repris et que celui ci travail avec 13 bandes, ce projet devra également travailler avec 13 bandes
    * Installation de snappy (librairie SNAP en python) et réalisation de test 
* **Prise en main QGIS3 - 30min**
    * Installation et prise en main
    * Regroupement de différentes images de Sentinel-2 dans une seule image

## Semaine 2 - 01.03.2021 - 05.03.2021 [12h00]
* **Test resampling image au format (10980,10980) - 1h30**
    * Installation OpenCv2
    * Recherche information methode de resampling utilisé par Sentinel 2 et Eurosat
    * Resampling des images avec une interpolation bicubic
* **Test normalisation des images jp2 - 1h00**
    * Recherche informations à propos du type jp2
    * Recherche informations à propos du format des image Sentinel (unint16)
    * Normalisation des images Sentinel
    * Note : il s'est averé par la suite que cette méthode de normalisation n'est pas la même qu'utilisé pour les images Eurosat. Les images Eurosat sont normaliss en soustrayant la moyenne de chaque bande et en divisant par l'écart type de chaque bande (sur tout le dataset Eurosat)
* **Test découpage image en plusieurs parties - 1h30**
    * Découpage des images en plusieurs parties de taille uniforme (64 par 64 pour l'instant)
    * Processing des images pour obtenir le même format que le réseau de neuronne qui sera utilisé pour le transfer learning
* **Implémentation script d'exportation des petites images sur le disque - 2h00**
    * Cette opération est effectué dans le but de pouvoir utiliser un generateur Python
    * Note : les images ne doivent pas être ecrite sur le disque normalisé sinon elle ne sont pas valide selon le format TIFF.
* **Lecture documentation transfer learning et Keras DataGenerator - 0h30**
    * Le DataGenerator peut être utile pour charger les images par lots et pas tout en même temps dans la mémoire
    * Lecture article à propos du transfer learning
* **Création d'un générateur pour charger les images par petits lots + normalisation à la volée - 2h30**
    * Note : Le generateur par défaut de Keras ne permet pas de charger des images avec plus de 4 canaux, il a donc fallu ecrire un generateur personnalisé
    * Création du script pour chercher les images sur le disques et effectuer la séparation entre le jeu d'entrainement et de test
    * Normalisation des images a la volée selon la méthode expliqué plus haut (moyenne et ecart-type par bande)
* **Mise en place d'un réseau de neuronne basique - 2h00**
    * Mise en place d'un réseau de neuronne (DenseNet) en mode transfert learning pour tester la pipeline complete d'entrainement
    * Note : la pipeline fonctionne correctement avec le générateur et le réseau s'entraine correctement. 
* **Prise en main des fichiers Shapefile - 1h00**
    * Reception des fichiers Shapefile contenant les coordonnées dans champs de cafés
    * Lecture et prise en main des scripts de traitement des shapesfiles (script du travail de bachelor)

## Semaine 3 - 08.03.2021 - 12.03.2021 [12h00]
* **Mise en place des scripts de traitements de shapefile - 1h00**
    * Test et réutilisation des script permettant de :
        * Recupérer la classe du shapefile
        * COnversion des différents système de coordonnées (fichier raster, latitude/longitude)
        * Attribuer à chaque petite image une positon par rapport à la grande image
    * Script pour la création de l'arborsence des images sur le disque (classes)
    * Modification du script d'exportation des images pour definir la bonne position geographique (petite image)
    * Modification du script d'exportation des images pour enregister les images dans le bon dossier (gestion images multilabel et sans label)
* **Deroulement du script qui attribue une localisation à chaque petite image - 2h30**
    * Comme il s'agit d'une étape clé du projet (labelisation des images), j'ai pris du temps pour verifier que la bonne position géographique était attribué au images
    * Renseignement sur le fonctionnement des différents système de coordonnées (CRS, EPSG, latitude longitude)
    * Verification sur une carte que les coordonnées était bien généré au bonne endroit
    * Verification de la cohérence de la labélisation des images (en vérifiant manuellement)
* **Telechargement des mêmes images que le travail de bachelor - 0h30**
    * Selection des mêmes 5 images que le travail de bachelor
    * Verification dans QGIS
* **Amélioration du script d'importation des bands dans le notebook - 1h00**
    * Amélioration pour selectionner automatiquement chaques bands à partir des zip de Sentinel
    * Modification des anciens script pour utiliser ``os.path.join()`` pour la portabilité
* **Mise en place du projet sur les serveurs de la HEIG-VD - 2h30**
    * Installation du VPN
    * Connexion sur le serveur, mise en place environnement virtuel et installation des packages
    * Mise en place jupyter sur un navigateur distance
    * Chargement des images sentinel
* **Lancement su script pour prétraité et découper les images sur le serveur - 0h30**
    * Une erreur est survenue car les bands n'était pas retourné dans le même ordre sur Linux que sous Windows
    * Correction du problème et lancement du script
* **Lancement du premier modèle sur le serveur - 1h00**
    * Modèle de transfer learning à partir d'un modèle trouvé sur tfhub (https://tfhub.dev/google/remote_sensing/eurosat-resnet50/1)
    * Champs de café vs le reste
    * Note : l'entrainement est très lent (360 secondes par epoch)
* **Recherche pour l'acceleration de l'entrainement du modèle sur le serveur - 1h30**
    * Recherche de la présence d'une carte graphique sur le serveur
    * Une carte graphique **AMD** se trouve sur le serveur, cepandant  la compatibilité est plus faible pour faire tourner des libraires de ML sur les cartes AMD que NVIDIA
    * Une solution a été trouvé et mise en place pour tester l'utilisation de la carte graphique avec Keras (https://plaidml.github.io/plaidml/)
    * Cette méthode fonctionne correctement avec Keras, mais comme le modèle utilise une couche tensorflow la mise en place avec cette libraire a été interompu
    * La mise en place de solution entre les cartes graphiques MAD et tensorflow nécessite les droits d'admin sur le serveur
    * Note : pour le moment aucune solution n'a été trouvé pour ce problème
* **Installation de keras-gpu sur ordinateur personnel - 1h30**
    * Carte graphique personnel : GTX960
    * Rechecher documentation sur keras-gpu pour windows
    * Installation keras-gpu avec anaconda
    * Création environnement virtuel
    * Tests pour verifier l'utilisation de la carte graphique lors de l'entrainement
    * Temps moyen de 60 secondes par epochs contre 360 sur le serveur (speedup * 6)

## Semaine 4 - 15.03.2021 - 19.03.2021 [12h00]
* **Entrainement de différent modèle sur ordinateur personnel - 3h30**
    * Entrainement d'un premier modèle :
        * Batch size = 32
        * Eurosat-resnet + Dropout (0.5) + Dense 128 + Dropout (0.5) + Dense 1
        * 50 epochs
    * Entrainement deuxième modèle :
        * Batch size = 32
        * Eurosat-resnet +  Dropout (0.5) + Dense 1024 +  Dropout (0.5) + Dense 256 +  Dropout (0.5) + Dense 1
        * 70 epochs
        * Data augmentation
    * Entrainement troisième modèle :
        * Batch size = 32
        * Eurosat-resnet + Dropout (0.5) + Dense 256 + Dropout (0.5) + Dense 1
        * 70 epochs
        * Data augmentation
        * Weighted class
        * learning rate = 0.001
    * 2 autres modèles ont été entrainés, cepandant un problème de séparation entre le jeu d'entrainement et de test fait qu'il sont inutilisable
* **Augmentation des données - 1h30**
    * Etant donnée que le dataset est assez pauvre en image (520 images de coffee), des techniques d'augmentation de données ont été mise en place
    * Les techniques suivantes ont été implémenté
        * Rotation aléatoire entre 0, 90, 180 et 270 degré
        * Inversion aléatoire des pixels verticalement ou horizontalement
    * Modification du DataGenerator pour effectuer ces transformations à la volée
* **Mise en place de Tensorboard - 0h30**
    * Installation de tensorboard pour comparer la performance entre les modèles
    * Comparaison de des différents modèles ci-dessus via tensorboard
* **Modèle culture vs Non culture - 3h00**
    * Le jeu de données peu-être séparé en 2 catégories les cultures (Cacao, poivre, caffé, riz, ...) et les non cultures (eau, ville, foret, ...)
    * L'idée est d'entrainé un modèle qui arrive a classifier une image comme étant une culture ou une non culture
    * Ecriture du script pour séparer les 2 classes, récupérer les labels et les chemin des images correspondantes
    * Modèle entrainé : 
        * Batch size = 32
        * Eurosat-resnet + Dropout (0.5) + Dense 128 + Dropout (0.5) + Dense 1
        * 70 epochs
        * Data augmentation
        * learning rate = 0.001
* **Rédaction du début du rapport - 1h00**
    * Redaction introduction, objectifs, sources
* **Problème tensorflow hub - 2h00**
    * Les modèles composés de la couche tensorflow hub ne donne pas le même resultats entre le modèle originalement entrainé et le modèle restoré après sauvegarde.
    * J'ai donc rechercher de la documentation sur ce problème mais sans trouver de solution, j'ai donc écrit un post sur StackOverflow : https://stackoverflow.com/questions/66662526/keras-model-with-tensorflow-hub-does-not-give-same-result-when-save-restore
    * J'ai cepandant trouver une solution consistant à sortir la couche tensorflow hub du modèle et effectuer la transformation fournit par cette couche en dehors du modèle.

## Semaine 5 - 22.03.2021 - 26.03.2021 [12h00]
* **Mise en place de la méthode de prédiction - 3h00**
    * Mise en place de la prédiction selon le TB de J.Rod
    * Création d'image en niveau de gris en fonction de la probabilité d'une image 64x64 d'être un champs de café ou non
    * Prediction des images bandes par bandes
* **Rédcation du rapport - 1h30**
    * Ecriture Etat de l'art
    * Etat du projet (ancien TB, ancien TM)
    * Prétraitement des images
    * Générateur et augmentation des données
    * Premier modèle avec tensorflow hub
* **Téléchargement de plus d'images pour entrainer les modèles - 2h00**
    * Actuellement il y a 5 images correspondant à la saison hiver, j'ai donc telecharger 5 nouvelles images pour les saisons restantes
    * Prétraitement des images et découpage
* **Entrainement des nouveaux modèles avec les nouvelles images - 0h30**
    * Entrainement modèle tensorflow hub coffee vs other
    * Entrainement modèle tensorflow hub culture vs no culture
* **Statistiques sur les images - 1h30**
    * Boxplot entre les différentes catégories d'image pour chaque bandes (Coffee, Rice, Urban, Water, Pepper, Nativege)
    * Heatmap moyenne des bandes pour chaques catégories ci-dessus
    * Histogramme des valeurs de chaques pixels de chaque bandes pour la class Coffee
* **Récupération des modèles de E.Ransom pour le transfer learning Coffee vs no coffee - 3h00**
    * Récupération des modèles hdf5 sur le serveur de la HEIG-VD
    * Importation du modèles suppression des couches top_level et ajout de couche Dense en sortie du modèle préentrainé pour le transfert learning
    * Mise en place de la pipeline avec le nouveau modèle
    * Entrainement du modèle
* **Entrainement en transfer learning culture vs no culture - 0h30**
    * Entrainement avec les modèles d'E.Ransom avec le transfer learning

## Semaine 6 - 29.03.2021 - 02.04.2021 [14h00]
* **Découpage des images en 32x32 - 0h30**
* **Entrainement des modèles avec les images 32x32 - 2h00**
    * Entrainement modèles Coffee vs no coffee
    * Entrainement modèles culture vs no culture
* **Lecture de l'article sur Fastaii et tentative de mettre en place l'environnement - 3h00**
    * Lecture de l'article https://link.medium.com/ncZaHz2YLeb sur la classification d'image Satellite avec FastAI
        * Ce qui ressort de l'article est que de bons resultat sont obtenus en utilisant pas toute les 13 bands mais en effectuant des variations de bandes.
    * Tentative de mise en place du même environement que dans l'article avec FastAI, cepandant de nombreux problème ont été rencontrés :
        1. La version du tutoriel n'est plus la meme que la version actuelle et la librairie a beaucoup changer entre temps
        2. Le librairie a très peu de documentation, ce qui ne facilite pas sa mise en place
    * Après plusieurs tentatives cette solution a été mise en pause pour ne pas perdre trop de temps.
* **Affichage de graphique de type GradCAM et FilterMaximisation - 3h00**
    * Dans le but de visualiser quelle partie de l'image sont utilisées pour determiner la classe lors de la classification, des graphiques de type GradCam et FilterMaximisation ont été mis en place.
    * On constate que pour les images de caffé, les fleurs blanches ont l'air d'être prise en compte pour la classification, pour la foret vierge on constate que les arbres denses sont pris en compte pour la classification.
* **Entrainement des modèles de transfer leanrning (E.Ransome) en modifiant le nombre de bande - 1h00**
    * Entrainement du modèle en conservant uniquement les bandes de haute résolution (1,2,3,4,5,6,7,8,11,12)
    * Modification de l'input du modèle pour accepter un nombre de bande < 13
    * Le modèle n'obtient pas du tout de bon résultat car l'ordre des bandes n'est pas conservé dans cette version
    * Cette méthodologie est donc a abandonné.
* **Entrainement des modèles de transfer learning (E.Ransome) en modifiant la taille des images d'entrée (32x32) - 1h30**
    * Modification de l'input du modèle pour accepter les images de taille (32x32)
    * Des résultats satisfaisant sont obtenus mais quand même moins bon qu'avec des images de 64x64
    * L'entrainement a été réalisé pour Coffee vs Other, Culture vs No Culture
* **Comparaison résultats modèles 32x32 vs 64vs64 - 1h00**
    * Pour les modèles avec Tensorflow hub et le transfer learning (E.Ransome)
    * Création de grahpique pour comparer la loss et l'accuracy
    * A première vue sur les deux types de modèles (coffee et culture) les images 64x64 semblent donner de meilleur résultats
* **Entrainement de modèles de transfer learning (E.Ransome) en ajoutant une couche de pooling - 0h30**
    * Ajout d'une couche de pooling immédiatement après la sortie du modèle de transfer learning.
    * Des mauvais moins bon résultats ont été obtenu qu'avec la couche Flatten() utilisé par tout les autres modèles jusqu'à présent.
    * Cette soltuion a donc été mise en suspens pour le moment.
* **Ajout d'une couche de pooling pour modifier le nombre de bandes en entrée et entrainement des modèles coorespondant- 4h00**
    * Ajout d'une couche de pooling "trainable" pour modifier conserver uniquement le nombre de channel désiré.
    * Plusieurs configurations ont été mise en place :
        * Input 10 bandes -> Conv2D(3,(1,1)) -> Modèle transfer learning RGB (3 bandes)
    * Entrainement des modèles correspondants

# Idées
* Autoencoder
* Reduction de dimension 13 -> 3

# Taches
* Cross val
* Entrainement image une seule saison
* Preprcess avec mean et std jeu de données

# Informations
* Vietnam saison :
    * Saison seche : novembre - avril
    * Saison humide : mai - octobre
* Saison occidentale :
    * Printemps : 20 Mars - 21 Juin
    * Ete : 21 Juin - 21 Septembre
    * Automne : 22 Septembre - 20 Decemembre
    * Hiver : 21 Decembre - 19 Mars
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

## Semaine 6 - 29.03.2021 - 02.04.2021 [15h00]
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
* **Ajout d'une couche de pooling pour modifier le nombre de bandes en entrée et entrainement des modèles coorespondant [RGB]- 4h30**
    * Ajout d'une couche de pooling "trainable" pour modifier conserver uniquement le nombre de channel désiré.
    * Plusieurs configurations ont été mise en place :
        * Input 10 bandes haute résolution (1,2,3,4,5,6,7,8,11,12) 
            * Conv2D(3,(1,1)) -> Modèle transfer learning RGB (3 bandes)
        * Input 13 bandes  
            * Conv2D(3,(1,1)) Modèle transfer learning RGB (3 bandes)
        * Input 6 bandes RGB + SWIR 
            * Conv2D(3,(1,1)) -> Modèle transfer learning RGB (3 bandes)
    * Entrainement des modèles correspondant en utilisant le modèle RGB préentrainé de E.Ransome.
    * Conclusion : Les résultats obtenu sont inférieurs à la méthode traditionnel 13 bandes de transfer learning. Pour l'expérience utilisé on constate que moins le modèle utilise de band plus les résultats obtenu sont meilleurs (mais toujours moins bon que l'expérience traditionnelle). 

## Semaine vacances - 05.04.2021 - 11.04.2021 [11h00]
* **Ajout d'une couche de pooling pour modifier le nombre de bandes en entrée et entrainement des modèles coorespondant [13bands]- 3h00**
    * Même expérience que préccedement mais utilisation du modèle pré-entrainé 13 bandes au lieu de RGB.
    * Plusieurs configurations ont été mise en place :
        * Input 10 bandes haute résolution (1,2,3,4,5,6,7,8,11,12) 
            * Conv2D(13,(1,1)) -> Modèle transfer learning RGB (3 bandes)
        * Input 6 bandes RGB + SWIR 
            * Conv2D(13,(1,1)) -> Modèle transfer learning RGB (3 bandes)
    * Note : Encore une fois, cette technique n'a pas amélioré les résultats par rapport à la méthode origniale utilisant les 13 bands
* **Mise en place de graphes et stats pour comparer le résultats des deux expérience de vraitions du nombre de bandes - 2h00**
* **Entrainement modèle Transfer learning avec images du printemps - 3h00**
    * D'après l'etude réalisé par S.Walther[1], deux informations importantes sont mise en avant :
        1. Les mois de Janvier à Avril sont les mois de l'années au Vietnam ou il y a le moins de couverture nuageuse sur les images satellite. Ceci va donc permettre de limiter la précense d'image inutilisable (car rempli de nuage )dans le jeu de données.
        2. De Janvier à Avril les arbres à Café sont fortement arrosé, dans le but de fleurir en Février. Durant cette période, les arbres a café sont donc plus facilement visible depuis le ciel à l'aide des flurs blanches qui les composent. 
    * A partir de ces 2 infos, les images de l'année 2021 ont été récupéré sur cette période, découpé et prétraité. 
    * Les 2 modèles de transfer learning 13 bands ont été entrainé avec ces nouvelles images. 
* **Entrainement des modèles de transfer learning en modifiant les paramètres de normalisation - 3h00**
    * Jusqu'a présent dans le but de reproduir fidélement le preprocessing effectué par E.Ransome, une normalisation de type z-norm avec la moyenne et l'écart type de toute les images d'Eurosat a été utilisé. En appliquant cette normalisation sur nos images on constate que les pixels ne sont pas centrés sur la valeur 0 (ce qui est le but d'une normalisation z-norm). Ceci provient surement du fait que les images Eurosat capturé sur des paysages européen de présente pas les mêmes caractéristiques que les images du sol vietnamien.
    * L'idée est alors de calculer la moyenne et l'écart type sur les images du Vietnam et d'utiliser ces paramètres pour effectuer la normalisation z-norm sur ces parametres et de réentrainer les modèles.
    * Etape :
        1. Calcul de la moyenne et std de toutes les images sur toutes une année (100000 pixels aléatoire par images ont été selectionnés)
        2. Réalisation d'histogramme pour comparer la répartition des pixels entre les params eurosat et vietnam
        3. Entrainement des nouveaux modèles
        4. Comparaison et analyse des résultats

## Semaine 7 - 12.04.2021 - 18.04.2021 [12h00]
* **Fin d'entrainement des modèles avec les nouveaux paramètres du vietnam et analyse des résultats**
* **Rédaction du rapport - 4h00**
    * Normalisation
    * DenseNet Transferlearning
    * Variation de bande
    * Culture vs non culture
    * Ajout de schéma pour le préprocessing, la variation de bande et l'architecture des différents modèles
* **Installation du setup sur Google Colab - 2h00**
    * Installation des notebooks et telechargement des images
    * Un problème est survenu car les images était charger depuis le drive et le chargement des images lors de l'entrainement du modèle était très lent, il a donc fallu telecharger les images directement dans le notebook et les unziper a la volée
    * Entrainement du modèle de transfer learning en validation croisée
* **Installation du setup sur le serveur trex - 1h00**
    * Telechargement du projet 
    * Création de l'environnement virtuel
    * Accès au GPU du serveur
    * Telechargement des images sur le serveur
* **Validation des modèles avec de la cross validation - 5h00**
    * Les modèles suivants ont été entrainé avec de la cross validation sur 5 folds :
        * Modèle transfer learning, Coffee vs Other, params eurosat
        * Modèle transfer learning, Coffee vs Other, params vietam
        * Modèle transfer learning, Coffee vs Other, spring image, params vietam
        * Modèle transfer learning, Culture vs non-culture, params eurosat

## Semaine 8 - 19.04.2021 - 25.04.2021 [12h00]
* **Finetuning du modèle - 1h30**
    * Création du notebook pour récupérer un modèle sauvegarder et le definir comme "trainable"
    * Lancement de l'entrainment du modèle avec un petit learning sur 150 epoch
    * Conclusion : Le finetuning du modèle n'améliore pas les résultats du prèccedent modèle
* Note : une modification du serveur trex de la Heigvd, a fait que le serveur était indisponible pendant une certaine période. Une fois la connection au serveur réussi les drivers de la carte graphique n'était plus à jour, il a donc fallu déplacer une partie du travail réaliser sur Google Colab. Ces différentes étapes ont pris **1h00** pour être réalisé
* **Recherche de documentation et mise en place de cross validation spatiale - 2h30**
    * La librairie ![https://github.com/SamComber/spacv](spacv) proposé par M.Satizabal Mejia permet d'effectuer une cross validation spatiale. C'est à dire qu'il est possible d'entrainer le modèle sur une certaine zone géographique et de le tester sur une autre.
    * Installation et prise en main de spacv, geopandas et shapely
    * Création d'un notebook de test qui permet de visualiser sur une carte du vietnam les différents points qui serait selectionner pour une cross validation spatiale. 
    * Visualisation également de toutes les labels disponibles avec leur classes correspondantes sur une carte du Vietnam.
* **Selection d'un nouveau jeu de données et découpage géographique - 3h30**
    * Il a été discuté lors d'une séance que de prendre plusieurs fois une image sur une période de l'année de l'année est une mauvaise idée pour plusieurs raisons :
        * Premièrement il est plus difficile pour le modèle de de généraliser sur toutes les images de toutes une année. Au vu des différents résultats obtenus, les images du de la période Janvier - Avril sont donc selectionné.
        * Comme une même image avec des différences très minime peut se retrouver plusieurs fois dans le jeu de données, il se peut que le modèle rencontre des images très semblable durant l'entrainemen, la validation et le test ce qui peut expliquer les très bonne performance obtenu sur certain modèle. 
    * Telechargement de nouvelles images pour la période de Janvier à Avril (une image satelite sur toute cette période)
    * Modification du notebook permettant de découper les images pour adapter au découpage géographique: 
        * Comme cette fois ci on veut garder une correspondance entre l'image exporté et sa localisation, un dataframe est alors créer avec comme information, le chemin de l'image sur le disque, le label correspondant et sa position géographique. 
        * Une fois toutes les images exportés sur le disque, une première sépartion géographique du dataframe est effectuée pour séparé le jeu d'entrainement et de test.
        * Le jeu d'entrainement et de test sont exportés sur le disque au format CSV dans le but d'avoir les mêmes jeux de données pour les différentes expérience. 
    * Lancement du notebook de découpage des images
* **Entrainement des modèles avec la cross validation spatiale pour Café- 3h30**
    * L'entrainement des modèles suivantes a été relancé avec la cross validation spatiale :
        * Transfer learning DenseNet 64x64
        * Transfer learning DenseNet 32x32
        * Transfer learning DenseNet 64x64 Vietnam
        * Transfer learning DenseNet 32x32 Vietnam

## Semaine 9 - 26.04.2021 - 02.05.2021 [12h00]
* **Entrainement des modèles avec la cross validation spatiale Culture - 2h00**
    * L'entrainement des modèles suivantes a été relancé avec la cross validation spatiale :
        * Transfer learning Densenet 64x64
        * Transfer learning Densenet 32x32
* **Analyse des résultats de la cross val spatial - 2h30**
    * On constate que les résultats sont inférieurs avec la cross validation spatiale.
    * Différente analyse ont été réalisé il s'avere que la séparation du jeu de données n'est pas stratifié, c'est a dire que pour certaine fold on se retrouve avec un très petit nombre de champs de café. En effet les champs de café ne se trouve pas equitablement réparti géographiquement ce qui conduit a ce problème. Le modèle n'arrive donc pas bien à généraliser et donne des résultats très différent d'une epoch à l'autre.
    * Une autre technique dois etre trouvé pour avoir une répartion plus equitable entre les classes pour chaque fold (et egalement entre le jeu d'entrainement et de test)
* **Entrainement des différentes expériences réalisé jusqu'ici avec un jeu de données représentatif - 5h00**
    * Les jeux de données utilisés jusqu'a présent (toute l'annnée et 5 images du printemps) ne sont pas représentatif car une image semblable peut se trouver dans le jeu d'entainement et de test, ce qui fausse les résultats.
    * Il a donc été décidé de créer un nouveau jeu d'entrainement sur la période du printemps avec une seule image par label. La séparation entre le jeu d'entrainement et de test a été créé de manière stratifié. Le jeux de données ont été enregistré sur le disque pour pouvoir reproduire les expériences d'un modèle a l'autre.
    * Les modèles suivants ont été ré-entrainé :
        * Café :
            * Transfer learning DenseNet 64x64
            * Transfer learning DenseNet 32x32
            * Transfer learning DenseNet 64x64 Vietnam
            * Transfer learning DenseNet 32x32 Vietnam
            * Tensorflow hub DenseNet 64x64
            * Tensorflow hub DenseNet 32x32
        * Culture :
            * Transfer learning DenseNet 64x64
            * Transfer learning DenseNet 32x32
            * Tensorflow hub DenseNet 64x64
            * Tensorflow hub DenseNet 32x32
* **Analyse des résultats des résultats de l'entrainement des modèles sur le nouveau jeu de données - 2h30**
    * Création de boxplot et histogramme pour comparer les différentes métriques des modèles (accuracy, f1-score, loss)
    * Les constatations suivantes peuvent être faites :
        * Caffé : 
            * Les modèles DenseNet performent mieux que les modèle ResNet 3 bandes (tensorflow hub)
            * Les modèles avec des images 64x64 donnent legerement de meilleur résultat que les 32x32
            * L'utilisation des paramètres du Vietnam n'améliore pas les perf par rapport au paramètre classique
            * Le modèle qui performe le mieux est le modèle DenseNet 64x64 avec les params Eurosat
        * Culture : 
            * Les modèles DenseNet performent mieux que les modèle ResNet 3 bandes (tensorflow hub)
            * Les modèles avec des images 64x64 donnent legerement de meilleur résultat que les 32x32 
            * Le modèle qui performe le mieux est le modèle DenseNet 64x64 avec les params Eurosat
            * Les modèles donnent de meilleur performance pour classifier les cultures vs non cultures que Café contre reste

## Semaine 10 - 03.05.2021 - 09.05.2021 [12h00]
* **Rédaction du rapport - 4h00**
    * Reformulation introduction
    * Jeu de données (Stratifié vs spatial)
    * Focal loss
    * Résultats (cross val, spatial, café, culture, variation de bande)
    * Discussion (café, culture, variation)
* **Cross validation spatial stratifié - 4h00**
    * Création d'un script pour effectuer une cross validation spatiale de manière stratifé, un equilibrage est fait en recupérant les points les plus proches dans le jeu d'entrainement et et les placants dans le set de validation ou inversement quand cela est nécaissaire. Ceci est fait pour les 2 classes.
    * Test de l'algorithme implémenté en affichant les points sur la carte
    * Entrainement de modèle avec cette nouvelle séparation. 
    * Conclusion : cette stratégie n'a pas permis d'améliorer les résultats par rapport à la cross validation spatial standard. De plus la séparation stricte entre la zone de validation et d'entrainement n'est plus conservé, ce qui est un des buts principaux de l'utilisation de spacv.
* **Cross valiation spatial par classe - 3h00**
    * Dans l'idée de garder un nombre de point similaire par classe pour chaque fold la cross validation spatiale est efftcué pour chacune des classes (coffee/other) et ensuite les 2 ensemble sont mis ensemble.
    * Test de l'algorithme implémenté en affichant les points sur la carte
    * Entrainement de modèle avec cette nouvelle séparation. 
    * Conclusion : cette stratégie n'a pas permis d'améliorer les résultats par rapport à la cross validation spatial standard. On constate encore beaucoup de variabilité dans les scores d'une fold à l'autre
* **Cross valiation spatial par classe sur 10 fold et mise en commun - 1h00**
    * Actuellement la séparation train/val est très strict, le modèle est entrainé sur une zone géographique très différente de laquel il est validé. Les images peuvent être très différentes d'une zone à une autre notamment au niveau du climat et du type de sol. Une idée est alors d'effectuer le même algo que préccedement mais sur 10 fold et d'ensuite réunir les folds 2 par 2 ensemble pour en obtenir par exemple 5. Ainsi le jeu de validation sera réparti sue au moins 2 zone différente de la carte.
    * Test de l'algorithme implémenté en affichant les points sur la carte
 
## Semaine 11 - 10.05.2021 - 16.05.2021 [12h00]
* **Rédaction du rapport - 4h00**
    * Ajout de contenu dans le chapitre : 
        * Résultats obtenus
        * Discussion
        * Réseau de neurone from Scratch
        * Séparation du jeu de données de manière spatiale
        * Scope out of scope
* **Entrainement des modèles de cross validation spatial (méthode 10 folds) - 2h00**
    * Entrainement avec les images de tailles 32x32 et 64x64
    * Méthode validé au vue des résultats obtenus, il faut maintenant reséparer le jeu de données pour évaluer sur le jeu de test
* **Comparaison des résultats entre les 4 méthodes de cross validation spatiale utilisé - 1h00**
    * Les 4 méthodes sont : 
        * Cross valisation spatiale classique : Beaucoup de variabilité entre les folds
        * Cross validation spatiale stratifié : peu de variabilité entre les folds, mais on perd le coté spatiale avec cett méthode, donc mise de coté
        * Cross validation spatiale par classe : Beaucoup de variabilité, similaire à la cross validation spatiale classique
        * Cross validation spatiale par classe (10 fold) : Varaibilité acceptable, il semble donc intéressant d'effectuer des recherches sur cette voix
* **Nouvelle séparation avec méthode des 10 folds - 2h30**
    * Nouvelle séparation du jeu de données avec la méthode de cross validation spatial pour le jeu d'entrainement, validation et test
    * Enregistrement des jeu de données sur le disque (pour chaque fold egalement)
    * Jeu de données enregistrés pour les cas suivants : 
        * Café contre reste 64x64
        * Café contre reste 32x32
        * Culture vs no-culture 64x64
        * Culture vs no-culture 32x32
* **Entrainement du modele DensetNet201 avec la nouvelle séparation de données en transfer learning - 2h30**
    * Entrainement pour : 
        * Café contre reste 64x64
        * Café contre reste 32x32
        * Culture vs no-culture 64x64
        * Culture vs no-culture 32x32

## Semaine 12 - 17.05.2021 - 23.05.2021 [12h00]
* **Entrainement du modele DensetNet201 avec la nouvelle séparation de données - from scratch - 2h30**
    * Reprise de la même architecture qu'utiliser préccedement mais sans les poids pré-entrainé
    * Entrainement pour : 
        * Café contre reste 64x64
        * Café contre reste 32x32
        * Culture vs no-culture 64x64
        * Culture vs no-culture 32x32
* **Comparaison des résultats entre modèle pré-entrainé ou from scratch - 2h00**
    * Comparaison des résultats obtenus pour la cross validation
    * Comparaison des résultats obtenus sur le jeu de test
    * Note : lors de l'entrainement les modèles entrainé from scratch semble obtenir de meilleur résultat que ceux de transfer learning, cependant sur le jeu de test on constate que les modèles de transfer learning performe mieux. 
* **Entrainement du modèle de transfer learning en fine tuning - 4h00**
    * Entrainement du modèle DenseNet64x64 13 bandes de transfer learning en mode fine tuning
    * Entrainement pour café contre reste et culture contre non-culture
    * Comparaison des résultats avec les modèles existants
* **Réarangement du code des notebooks - 1h30**
    * Parcours des différents notebooks et réarangement du code en vue du rendu
    * Ajout de commentaires
    * Suppresion des parties de code plus utilisé
    * Ajout d'un readme pour décrire les différents fichiers et notebooks du projets
* **Rédaction du rapport - 2h00** 
    * Ajout des nouuveaux résultats dans le rapport 
    * Ajout du chapitre répartition des classes

## Semaine 13 - 24.05.2021 - 30.05.2021 [12h00]
* **Entrainement de modèles multi-classe - 4h00**
    * Création du script pour séparer de manière spatiale les données multi classe
    * Création du script pour entrainer le modèle en multiclasse
    * Entrainement du modèle
    * Modification du script pour faire des prédictions avec un ensemble de modèle pour la classification multi classe
    * Comparaison des résultats obtenus avec les autres exprériences réalisé dans le projet
* **Entrainement de modèles multi-classe avec une sortie fixe - 4h00**
    * Dans le but de valider les résultats de la classification multi-classe l'idée est d'entrainer le modèle avec la sortie culture fixé à 0 et de constater si le modèle a de moins bon résultats ou non. 
    * Pareillement avec la sortie du café.
* **Ajout de la métrique Macro F1-Score et correction du F1-Score sur le test - 2h00**
    * Ajout de la métrique Macro F1-Score sur la prédiction des modèles pour le jeu de test
    * Pour la prédiction sur le jeu de test, le F1-Score était calculé sut la mauvaise classe -> correction du bug et mise a jour dans le rapport
* **Création d'un README qui liste les différents jeux de données utilisé - 1h00**
* **Rédaction du rapport - 1h00** 

## Semaine 13 - 31.05.2021 - 04.06.2021 [12h00]
* **Tri des modèles exportés et création README - 2h00**
* **Rédaction du rapport - 5h00**
    * Relecture du rapport
    * Rédaction conclusion, abstract
* **Finalisation du projet - 5h00**
    * Préparation des délivrables du projet
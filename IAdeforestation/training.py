import rasterio
import numpy as np
import random
from keras import backend as K

from tensorflow.keras.models import model_from_json

from IAdeforestation.preprocessing import normalize

eurosat_params = {'mean':[1353.036, 1116.468, 1041.475, 945.344, 1198.498, 2004.878, 2376.699, 2303.738, 732.957, 12.092, 1818.820, 1116.271, 2602.579],
                  'std':[65.479, 154.008, 187.997, 278.508, 228.122, 356.598, 456.035, 531.570, 98.947, 1.188, 378.993, 303.851, 503.181]}

vietnam_params = {'mean':[1279.534254, 1016.734146, 925.27579, 793.929164, 1073.835362, 1909.174038, 2299.416608, 2270.341238, 739.972412, 14.35029, 1872.530084, 1055.580112, 2581.31964],
                  'std':[217.06847657849937, 236.49447129038893, 254.3062726895201, 383.2039520109347, 368.1552150776269, 508.3797488499248, 648.9503962237852, 672.6241209212196, 250.63210405653427, 9.4263874644246, 805.0923719290897, 632.9663115986274, 746.86130213707]}

def generator(paths, classes, eurosat_mean, eurosat_std, batch_size=32, is_data_augmentation=True):
    
    while True : 
        random_indexs = random.sample(range(0,len(paths)), batch_size)
        batch_paths = paths[random_indexs]

        X = []
        Y = classes[random_indexs]

        for p in batch_paths:
            img = rasterio.open(p).read()
            img = np.float32(np.moveaxis(img, 0, -1))

            img = normalize(img, eurosat_mean, eurosat_std)
            
            if is_data_augmentation:
                
                # Angle rotation of 90, 180 or 270
                nb_rotation = random.randint(0,3)
                if nb_rotation != 0:
                    img = np.rot90(img,nb_rotation)
                    
                # Image flip
                if random.choice([True, False]):
                    img = np.flipud(img)
                    
                if random.choice([True, False]):
                    img = np.fliplr(img)
                
            X.append(img)
             
        X = np.asarray(X)
        yield (X,Y)

def generator_bands(paths, classes, eurosat_mean, eurosat_std, bands, batch_size=32, is_data_augmentation=True):
    
    while True : 
        random_indexs = random.sample(range(0,len(paths)), batch_size)
        batch_paths = paths[random_indexs]

        X = []
        Y = classes[random_indexs]

        for p in batch_paths:
            img = rasterio.open(p).read()[bands,:,:]
            img = np.float32(np.moveaxis(img, 0, -1))

            img = normalize(img, eurosat_mean, eurosat_std)
            
            if is_data_augmentation:
                
                # Angle rotation of 90, 180 or 270
                nb_rotation = random.randint(0,3)
                if nb_rotation != 0:
                    img = np.rot90(img,nb_rotation)
                    
                # Image flip
                if random.choice([True, False]):
                    img = np.flipud(img)
                    
                if random.choice([True, False]):
                    img = np.fliplr(img)
                
            X.append(img)
             
        X = np.asarray(X)
        yield (X,Y)

def keras_layer_generator(paths, classes, eurosat_mean, eurosat_std, keras_layer, batch_size=32, is_data_augmentation=True):
    
    while True : 
        random_indexs = random.sample(range(0,len(paths)), batch_size)
        batch_paths = paths[random_indexs]

        X = []
        Y = classes[random_indexs]

        for p in batch_paths:
            img = rasterio.open(p).read()[1:4,:,:]
            img = np.float32(np.moveaxis(img, 0, -1))

            img = normalize(img, eurosat_mean, eurosat_std)
            
            if is_data_augmentation:
                
                # Angle rotation of 90, 180 or 270
                nb_rotation = random.randint(0,3)
                if nb_rotation != 0:
                    img = np.rot90(img,nb_rotation)
                    
                # Image flip
                if random.choice([True, False]):
                    img = np.flipud(img)
                    
                if random.choice([True, False]):
                    img = np.fliplr(img)
                
            X.append(img)
             
        X = np.asarray(X)
        X_pre = keras_layer(X).numpy()
        yield (X_pre,Y)

def change_model(model, new_input_shape,custom_objects=None,verbose=False):
    # replace input shape of first layer
    
    config = model.layers[0].get_config()
    config['batch_input_shape']=new_input_shape
    model._layers[0]=model.layers[0].from_config(config)

    # rebuild model architecture by exporting and importing via json
    new_model = model_from_json(model.to_json(),custom_objects=custom_objects)

    # copy weights from old model to new one
    for layer in new_model._layers:
        try:
            layer.set_weights(model.get_layer(name=layer.name).get_weights())
            if verbose :
                print("Loaded layer {}".format(layer.name))
        except:
            print("Could not transfer weights for layer {}".format(layer.name))

    return new_model


def f1_score_keras(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def compute_score(geo_val, model):
    
    test_generator = generator(geo_val['path'].to_numpy(), 
                        geo_val['label'].to_numpy(), 
                        eurosat_params['mean'], 
                        eurosat_params['std'], 
                        batch_size=len(geo_val))
    
    model.evaluate(test_generator,steps=1)
    Y_true = []
    Y_pred = []
    for i in range (0,1):
        X, Y = next(test_generator)
        Y_pred.extend(np.where(model.predict(X) > 0.5, 1, 0))

        Y_true.extend(Y.tolist())

    Y_true = np.asarray(Y_true)
    Y_pred = np.asarray(Y_pred)
    cm = confusion_matrix(Y_true, Y_pred)
    print(cm)
    
    print(classification_report(Y_true, Y_pred))
    print(f"F1-Score : {f1_score(Y_true, Y_pred)}")
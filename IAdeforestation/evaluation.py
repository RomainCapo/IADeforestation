import numpy as np
import os
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, f1_score, classification_report, multilabel_confusion_matrix
from keras.models import load_model
from keras.layers import LeakyReLU
from focal_loss import BinaryFocalLoss
from keras import metrics

from IAdeforestation.tools import *
from IAdeforestation.training import *


def set_model_prediction(model_path, test_set, custom_objects, with_f1=False, labels=["Coffee", "Other"], title="title"):
    """Compute metrics of a set of models. Computed metrics are : Loss, Accuracy, F1-Score, Macro F1-Score.

    Args:
        model_path (string): path of folder that contains models.
        test_set (pandas.DataFrame): DataFrame with column path for images path and label.
        custom_objects (dict): Dict of custom objects to load with model.
        with_f1 (bool, optional): True if the model is train with F1-Score metrics, otherwise False. Defaults to False.
        labels (list, optional): Label for confusion Matrix. Defaults to ["Coffee", "Other"].
        title (str, optional): Title of confusion matrix. Defaults to "title".
    """
    test_generator =  generator(test_set['path'].to_numpy(), 
                            test_set['label'].to_numpy(), 
                            eurosat_params['mean'], 
                            eurosat_params['std'], 
                            batch_size=len(test_set))
    
    prediction_set = []
    evaluate = []
    
    X, y = next(test_generator)
    
    for path in os.listdir(model_path):
        if path.split(".")[1] == 'h5':
            restored_model = None
            if with_f1:
                restored_model = load_model(os.path.join(model_path, path), custom_objects, compile=False)
                restored_model.compile(optimizer=Adam(learning_rate=0.00001), loss='binary_crossentropy',metrics=[metrics.BinaryAccuracy(name='accuracy'),metrics.Precision(name='precision'),metrics.Recall(name='recall'),f1_score_keras])
            else : 
                restored_model = load_model(os.path.join(model_path, path), custom_objects)
            
            evaluate.append(restored_model.evaluate(test_generator, steps=1))
            prediction_set.append(np.where(restored_model.predict(X) > 0.5, 1, 0).reshape(-1).tolist())
            
    predictions = []
    for pred in zip(*prediction_set):
        predictions.append(np.argmax(np.bincount(pred)))
    
    cm = confusion_matrix(y, predictions)
    plot_confusion_matrix(cm, labels,title)
    
    if with_f1:
        losses, accs, precisions, recalls, f1 = zip(*evaluate)
    else :
        losses, accs, precisions, recalls = zip(*evaluate)

    print(f"Mean accuracy : {round(np.mean(accs),4)}")
    print(f"Stdev accuracy : {round(np.std(accs),4)}")
    print("\n")
    print(f"Mean loss : {round(np.mean(losses),4)}")
    print(f"Stdev loss : {round(np.std(losses),4)}")
    print("\n")
    print(f"F1-Score {labels[0]}: {round(f1_score(y, predictions, pos_label=0),4)}")
    print(f"F1-Score {labels[1]}: {round(f1_score(y, predictions, pos_label=1),4)}")
    print(f"Macro F1-Score : {round(f1_score(y, predictions, average='macro'),4)}")
    
def set_model_prediction_multi_label(model_path, test_set, custom_objects, with_f1=False):
    """Compute metrics of a set of models for multilabel model. Computed metrics are : Loss, Accuracy, F1-Score, Macro F1-Score.

    Args:
        model_path (string): path of folder that contains models.
        test_set (pandas.DataFrame): DataFrame with column path for images path and label.
        custom_objects (dict): Dict of custom objects to load with model.
        with_f1 (bool, optional): True if the model is train with F1-Score metrics, otherwise False. Defaults to False.
    """
    test_generator =  generator(test_set['path'].to_numpy(), 
                            test_set[['label_culture','label_coffee']].to_numpy(), 
                            eurosat_params['mean'], 
                            eurosat_params['std'], 
                            batch_size=len(test_set))
    
    prediction_set = []
    evaluate = []
    
    X, y = next(test_generator)
    
    for path in os.listdir(model_path):
        if path.split(".")[1] == 'h5':
            restored_model = None
            if with_f1:
                restored_model = load_model(os.path.join(model_path, path), custom_objects, compile=False)
                restored_model.compile(optimizer=Adam(learning_rate=0.00001), loss='binary_crossentropy',metrics=[metrics.BinaryAccuracy(name='accuracy'),metrics.Precision(name='precision'),metrics.Recall(name='recall'),f1_score_keras])
            else : 
                restored_model = load_model(os.path.join(model_path, path), custom_objects)
            
            evaluate.append(restored_model.evaluate(test_generator, steps=1))
            prediction_set.append(np.where(restored_model.predict(X) > 0.5, 1, 0))
            
    predictions = []
    for pred in zip(*prediction_set):
        culture_pred, coffee_pred = zip(*pred)

        predictions.append(np.array([np.argmax(np.bincount(culture_pred)),
                                    np.argmax(np.bincount(coffee_pred))]))
    
    cm = multilabel_confusion_matrix(y, predictions)
    
    plot_confusion_matrix(cm[0], ["Culture", "No-Culture"],"Confusion Matrix\nCulture vs No-Culture\nDenseNet 64x64")
    plot_confusion_matrix(cm[1], ["Coffee", "Other"],"Confusion Matrix\nCoffee vs other\nDenseNet 64x64")
    
    if with_f1:
        losses, accs, precisions, recalls, f1 = zip(*evaluate)
    else :
        losses, accs, precisions, recalls = zip(*evaluate)

    print("Global metrics")
    print(f"Mean accuracy : {round(np.mean(accs),4)}")
    print(f"Stdev accuracy : {round(np.std(accs),4)}")
    print("\n")
    print(f"Mean loss : {round(np.mean(losses),4)}")
    print(f"Stdev loss : {round(np.std(losses),4)}")
    print("\n")
    culture_pred, coffee_pred = zip(*predictions)
    culture_true, coffee_true = zip(*y)
    print("Culture vs no-culture")
    print(f"F1-Score Culture: {round(f1_score(culture_true, culture_pred, pos_label=0),4)}")
    print(f"F1-Score No culture: {round(f1_score(culture_true, culture_pred, pos_label=1),4)}")
    print(f"Macro F1-Score : {round(f1_score(culture_true, culture_pred, average='macro'),4)}")
    print(f"\n")
    print("Coffee vs other")
    print(f"F1-Score Coffee: {round(f1_score(coffee_true, coffee_pred, pos_label=0),4)}")
    print(f"F1-Score Other: {round(f1_score(coffee_true, coffee_pred, pos_label=1),4)}")
    print(f"Macro F1-Score : {round(f1_score(coffee_true, coffee_pred, average='macro'),4)}")

    
def compute_score(geo_val, model):
    """Compute metrics after each model fold. 

    Args:
        geo_val (pandas.DataFrame): DataFrame with column path for images path and label.
        model (keras.Model): Model to evaluate.
    """
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
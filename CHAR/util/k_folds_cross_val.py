from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn import metrics

def k_folds_x_val(model, X, y, k =  5): 
    results = cross_validate(model, X, y, cv=k, scoring=["accuracy"])
    mean_accuracy = results['test_accuracy'].mean()
    print(f"Mean Accuracy: {mean_accuracy}")
    return mean_accuracy, 

def get_cmat(model, X, y):
    y_pred = model.predict(X)
    return confusion_matrix(y, y_pred, normalize="pred")

def get_metrics(model, X, y):
    y_pred = model.predict(X)
    return accuracy_score(y, y_pred), f1_score(y, y_pred, average="weighted")
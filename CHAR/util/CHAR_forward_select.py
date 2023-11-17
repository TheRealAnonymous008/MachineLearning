from sklearn.feature_selection import SequentialFeatureSelector

# Initialize feature selector 

def forward_select(model, X, y, k): 
    sfs = SequentialFeatureSelector(estimator=model, n_features_to_select=k)
    sfs.fit(X, y)
    return sfs.get_feature_names_out(), sfs.transform(X)

def forward_select_and_fit(model, X_train, Y_train, k, X_test, Y_test):
    model : tree.DecisionTreeClassifier = skl.base.clone(model)
    _, Xt = forward_select(model, X_train, Y_train, k)
    model.fit(Xt, Y_train)

    # TODO: Run tests here
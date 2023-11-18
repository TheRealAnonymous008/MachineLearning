from sklearn.model_selection import GridSearchCV

def gridSearchHPO(model, search_space):
    grid_search = GridSearchCV(estimator=model,
                                param_grid=search_space,
                                scoring='accuracy',
                                cv=5,
                                verbose=3,
                                error_score='raise',
                                n_jobs=-1,  # -1 means max amount
                                )
    return grid_search
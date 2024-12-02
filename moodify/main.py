def train(model, X_train, y, preproc):
    model.fit(X_train)

    if preproc =='bert':
        preproc_lyrics_bert(X)

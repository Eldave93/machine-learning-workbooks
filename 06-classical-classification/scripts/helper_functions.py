# this creates the matplotlib graph to make the confmat look nicer
def pretty_confusion_matrix(confmat, labels, title, labeling=False, highlight_indexes=[]):
    import matplotlib.pyplot as plt
    import warnings
    labels_list = [["TN", "FP"], ["FN", "TP"]]
    
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            if labeling:
                label = str(confmat[i, j])+" ("+labels_list[i][j]+")"
            else:
                label = confmat[i, j]
            
            
            if [i,j] in highlight_indexes:
                ax.text(x=j, y=i, s=label, va='center', ha='center',
                        weight = "bold", fontsize=18, color='#32618b')
            else:
                ax.text(x=j, y=i, s=label, va='center', ha='center')
       
    # change the labels
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax.set_xticklabels(['']+[labels[0], labels[1]])
        ax.set_yticklabels(['']+[labels[0], labels[1]])

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    ax.xaxis.set_label_position('top')
    plt.suptitle(title)
    plt.tight_layout()
    
    plt.show()


def create_example_model(X, output):

    import pandas as pd
    from imblearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from imblearn.under_sampling import RandomUnderSampler


    log_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("sampler", RandomUnderSampler(random_state=123)),
        ("model", LogisticRegression(random_state=42))])

    y_train = X.loc[:,output]
    X_train = X.drop(output, axis=1)

    return log_pipe.fit(X_train, y_train)
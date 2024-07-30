from sample_generator import *
from figure_plot import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_blobs
from plotly.subplots import make_subplots

if __name__ == '__main__':
    df = get_samples()
    plot_data(df)



    # turn df into ndarray
    X = df.iloc[:, :2].values
    y = df.iloc[:, -1].values
    y = y.astype('int')

    # subplots 1
    fig_subplots = make_subplots(rows=1, cols=2)
    y_colors = []
    for i in np.arange(y.shape[0]):
        if y[i] == 0:
            y_colors.append('red')
        else:
            y_colors.append("blue")
    fig_subplots.add_trace(
        go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=y_colors)),
        row=1, col=1
    )

    # testing with inate fucitons
    # X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
    # df_new = pd.DataFrame(np.c_[X, y])
    # df_new.rename(columns={0: 'feature1', 1: 'feature2', 2: 'target'}, inplace=True)
    # plot_data(df_new)

    # model creation
    model = KNeighborsClassifier()
    model.fit(X, y)

    # model testing
    df_test = get_samples()
    X_test = df_test.iloc[:, :2].values
    y_test = df_test.iloc[:, -1].values
    y_test = y_test.astype('int')
    # X_test, y_test = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
    # df_test = pd.DataFrame(np.c_[X, y])
    # df_test.rename(columns={0: 'feature1', 1: 'feature2', 2: 'target'}, inplace=True)
    # df_test = pd.DataFrame(np.c_[X_test, y_test])
    # df_test.rename(columns={0: 'feature1', 1: 'feature2', 2: 'target'}, inplace=True)
    y_pred = model.predict(X_test)

    # subplots 2
    y_colors = []
    for i in np.arange(y_pred.shape[0]):
        if y_pred[i] == 0:
            y_colors.append('red')
        else:
            y_colors.append("blue")
    fig_subplots.add_trace(
        go.Scatter(x=X_test[:, 0], y=X_test[:, 1], mode='markers', marker=dict(color=y_colors)),
        row=1, col=2
    )
    fig_subplots.show()

    # plot preditions
    s_pred = pd.Series(y_pred)
    df_test['prediction'] = s_pred
    plot_test_data(df_test)

    plot_confussion_matrix(y_test, y_pred)

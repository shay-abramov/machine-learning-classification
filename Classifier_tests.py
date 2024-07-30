from sample_generator import *
from figure_plot import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_blobs

if __name__ == '__main__':
    df = get_samples()
    plot_data(df)

    # turn df into ndarray
    X = df.iloc[:, :2].values
    y = df.iloc[:, -1].values
    y = y.astype('int')

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

    # plot preditions
    s_pred = pd.Series(y_pred)
    df_test['prediction'] = s_pred
    plot_test_data(df_test)

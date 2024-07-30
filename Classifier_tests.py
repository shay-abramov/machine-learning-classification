from sample_generator import *
from figure_plot import *
from sklearn.neighbors import KNeighborsClassifier


if __name__ == '__main__':
    df = get_samples()
    plot_data(df)

    # turn df into ndarray
    X = df.iloc[:, :1].values
    y = df.iloc[:, -1].values

    # model testing
    model = KNeighborsClassifier()
    model.fit(X, y)
    df_test = get_samples()
    X_test = df.iloc[:, :1].values
    y_test = df.iloc[:, -1].values
    y_pred = model.predict(X_test)

    # plot preditions
    s_pred = pd.Series(y_pred)
    df['prediction'] = s_pred
    plot_test_data(df)

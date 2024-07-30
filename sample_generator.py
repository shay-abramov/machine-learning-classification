import numpy as np
import pandas as pd


def get_samples() -> pd.DataFrame:
    m1 = 60  # nuber of positive samples
    m2 = 20  # number of negative samples

    # positive sample generation
    base = np.linspace(200, 100, m1)
    noise = np.random.normal(0, 5, m1)
    base_with_noise = base + noise
    s1 = pd.Series(base_with_noise)
    base = np.linspace(240, 140, m1)
    noise = np.random.normal(0, 1, m1)
    base_with_noise = base + noise
    s2 = pd.Series(base_with_noise)

    df_positives = pd.DataFrame([s1, s2, np.ones(len(s1))]).T

    # negative sample generation
    base = np.linspace(40, 00, m2)
    noise = np.random.normal(0, 5, m2)
    base_with_noise = base + noise
    s1 = pd.Series(base_with_noise)
    base = np.linspace(50, 140, m2)
    noise = np.random.normal(0, 1, m2)
    base_with_noise = base + noise
    s2 = pd.Series(base_with_noise)

    df_negatives = pd.DataFrame([s1, s2, np.zeros(len(s1))]).T

    # combine and rename columns
    df = pd.concat([df_positives, df_negatives], axis=0)
    df.rename(columns={0: 'feature1', 1: 'feature2', 2: 'target'}, inplace=True)

    return df

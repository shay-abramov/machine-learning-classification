import numpy as np
import pandas as pd
import plotly.graph_objects as go


def plot_data(df: pd.DataFrame) -> None:
    negatives = df[df['target'] == 0]
    positives = df[df['target'] == 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=negatives['feature1'],
        y=negatives['feature2'],
        mode='markers',
        name='Negatives'
    ))
    fig.add_trace(go.Scatter(
        x=positives['feature1'],
        y=positives['feature2'],
        mode='markers',
        name='Positives'
    ))
    fig.show()


def plot_test_data(df: pd.DataFrame) -> None:
    TN = df[df['prediction'] == 0]
    FN = TN[TN['target'] == 1]
    TN = TN[TN['target'] == 0]
    TP = df[df['prediction'] == 1]
    FP = TP[TP['target'] == 0]
    TP = TP[TP['target'] == 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=TN['feature1'],
        y=TN['feature2'],
        mode='markers',
        marker_color='blue',
        marker_symbol='circle',
        name='TN'
    ))
    fig.add_trace(go.Scatter(
        x=FN['feature1'],
        y=FN['feature2'],
        mode='markers',
        marker_color='blue',
        marker_symbol='x',
        name='FN'
    ))
    fig.add_trace(go.Scatter(
        x=TP['feature1'],
        y=TP['feature2'],
        mode='markers',
        marker_color='red',
        marker_symbol='circle',
        name='TP'
    ))
    fig.add_trace(go.Scatter(
        x=FP['feature1'],
        y=FP['feature2'],
        mode='markers',
        marker_color='red',
        marker_symbol='x',
        name='FP'
    ))
    fig.show()

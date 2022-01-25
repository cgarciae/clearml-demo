from typing import Tuple

import clearml
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import typer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
import seaborn as sns

titanic_url = (
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)
label = "survived"
excluded_columns = {"passengerid", "name", "ticket"}
column_groups = {
    "numerical": [
        "age",
        "fare",
    ],
    "categorical": [
        "pclass",
        "sibsp",
        "parch",
        "ticket",
        "name",
        "sex",
        "passengerid",
        "cabin",
        "embarked",
    ],
    "label": [label],
}


def get_dataset():
    df = pd.read_csv(titanic_url)
    df.rename(columns={col: col.lower() for col in df.columns}, inplace=True)

    return df


def get_transformer():
    return ColumnTransformer(
        [
            (
                "numerical",
                Pipeline(
                    [
                        (
                            "scale",
                            StandardScaler(),
                        ),
                        (
                            "impute",
                            SimpleImputer(
                                strategy="mean",
                                add_indicator=True,
                            ),
                        ),
                    ]
                ),
                [
                    col
                    for col in column_groups["numerical"]
                    if col not in excluded_columns
                ],
            ),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="NA",
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse=False),
                        ),
                    ]
                ),
                [
                    col
                    for col in column_groups["categorical"]
                    if col not in excluded_columns
                ],
            ),
            (
                "label",
                OrdinalEncoder(),
                column_groups["label"],
            ),
        ]
    )


def get_experiment_tracker(project_name: str) -> clearml.Task:

    task = clearml.Task.init(
        project_name=project_name,
        task_name=project_name,
        reuse_last_task_id=False,
    )

    # useful for multi-task pipelines
    # task = clearml.Task.create(
    #     project_name,
    #     task_name=project_name,
    #     task_type="training",
    #     add_task_init_call=True,
    # )
    # task.mark_started()
    # task.started()

    return task


def make_plots(df: pd.DataFrame, tracker: clearml.Task):

    for col in column_groups["numerical"]:
        title = f"{col} distribution"
        fig = plt.figure(figsize=(16, 10))
        plt.hist(df[col], bins=20)
        tracker.logger.report_matplotlib_figure(
            title=title,
            series=title,
            figure=fig,
            iteration=0,
        )

    # bar chart for categorical variables
    for col in column_groups["categorical"]:
        if col in excluded_columns or col == "cabin":
            continue

        title = f"{col} distribution"
        fig = plt.figure(figsize=(16, 10))
        df[col].value_counts().plot(kind="bar")
        tracker.logger.report_matplotlib_figure(
            title=title,
            series=title,
            figure=fig,
            iteration=0,
        )


def split_data(
    df: pd.DataFrame, train_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(df, train_size=train_size, random_state=42)
    return train, test


def get_model(
    n_features: int, n_classes: int, n_layers: int, n_units: int
) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(128, activation="relu", input_shape=[n_features])
            for _ in range(n_layers)
        ]
        + [tf.keras.layers.Dense(1, activation="sigmoid")]
    )

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )

    return model


def log_history(history: tf.keras.callbacks.History, tracker: clearml.Task):
    for key in history.history.keys():
        for i, value in enumerate(history.history[key]):
            tracker.logger.report_scalar(
                key,
                series=key,
                iteration=i,
                value=value,
            )


def report_scalar(tracker: clearml.Task, key: str, value: float):
    tracker.logger.report_scalar(
        key,
        series=key,
        iteration=0,
        value=value,
    )
    tracker.logger.report_scalar(
        key,
        series=key,
        iteration=1,
        value=value,
    )


def main(
    project_name: str = typer.Option("titanic-demo", help="Name of the project"),
    train_size: float = typer.Option(0.9, help="Size of the training set"),
    epochs: int = typer.Option(100, help="Number of epochs"),
    batch_size: int = typer.Option(32, help="Batch size"),
    n_layer: int = typer.Option(2, help="Number of layers"),
    n_units: int = typer.Option(32, help="Number of units"),
):
    tracker = get_experiment_tracker(project_name)

    # log hyperparameters
    tracker.connect(
        {
            "train_size": train_size,
            "epochs": epochs,
            "batch_size": batch_size,
            "n_layer": n_layer,
            "n_units": n_units,
        }
    )

    df = get_dataset()
    train_df, test_df = split_data(df, train_size=train_size)
    make_plots(train_df, tracker)

    transformer = get_transformer()
    transformer.fit(train_df)
    X_train = transformer.transform(train_df)
    X_test = transformer.transform(test_df)

    X_train, y_train = X_train[:, :-1], X_train[:, -1].astype(np.int32)
    X_test, y_test = X_test[:, :-1], X_test[:, -1].astype(np.int32)

    model = get_model(
        n_features=X_train.shape[1],
        n_classes=len(np.unique(y_train)),
        n_layers=n_layer,
        n_units=n_units,
    )

    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
    )

    log_history(model.history, tracker)

    y_pred = model.predict(X_test)

    # report metrics
    f1 = f1_score(y_test, y_pred.round())
    accuracy = accuracy_score(y_test, y_pred.round())
    precision = precision_score(y_test, y_pred.round())
    recall = recall_score(y_test, y_pred.round())
    cm = confusion_matrix(y_test, y_pred.round())

    report_scalar(tracker, "f1_final", f1)
    report_scalar(tracker, "accuracy_final", accuracy)
    report_scalar(tracker, "precision_final", precision)
    report_scalar(tracker, "recall_final", recall)

    # report confusion matrix
    fig = plt.figure(figsize=(16, 10))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    tracker.logger.report_matplotlib_figure(
        title="Confusion matrix",
        series="Confusion matrix",
        figure=fig,
        iteration=0,
    )

    # report precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred)
    fig = plt.figure(figsize=(16, 10))
    plt.plot(thresholds, precisions[:-1], color="blue", label="Precision")
    plt.plot(thresholds, recalls[:-1], color="green", label="Recall")
    plt.legend()
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    tracker.logger.report_matplotlib_figure(
        title="Precision-Recall curve",
        series="Precision-Recall curve",
        figure=fig,
        report_interactive=False,
    )
    # plt.show()


if __name__ == "__main__":
    typer.run(main)

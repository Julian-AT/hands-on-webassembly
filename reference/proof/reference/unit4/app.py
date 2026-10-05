from pathlib import Path
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from shinywidgets import render_plotly
from shinywidgets import output_widget
import plotly.express as px
from plotly.subplots import make_subplots

from shiny import App, render, ui, reactive

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# ------------------------- Global settings -----------------------------------

sns.set_theme(style="whitegrid")


# ------------------------- Datasets ------------------------------------------

def load_iris() -> pd.DataFrame:
    """
    Load iris dataset [1].

    Classes: 0 = setosa, 1 = versicolor, 2 = virginica

    [1] R.A. Fisher, "The use of multiple measurements in taxonomic problems",
        Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to Mathematical Statistics",
        John Wiley, New York, 1950.

    :return: iris dataset
    """
    iris_data = datasets.load_iris()
    data = pd.DataFrame(iris_data['data'], columns=iris_data['feature_names'])
    data['species'] = iris_data['target']
    return data
    

def load_wine() -> pd.DataFrame:
    """
    Load wine dataset [1].

    [1] Forina, M. et al, PARVUS - An Extendible Package for Data Exploration, Classification and Correlation.
        Institute of Pharmaceutical and Food Analysis and Technologies, Via Brigata Salerno, 16147 Genoa, Italy.

    :return: wine dataset
    """
    wine_data = datasets.load_wine()
    data = pd.DataFrame(wine_data['data'], columns=wine_data['feature_names'])
    data['cultivator'] = wine_data['target']
    return data

def load_pima():
    """
    Load Pima Indians Diabetes dataset [1].

    [1] https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
    :return: Pima Indians Diabetes dataset
    """
    import pandas as pd
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    cols = ["Preg", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DPF", "Age", "Outcome"]
    df = pd.read_csv(url, names=cols)
    return df

def load_banknote():
    """
    Load banknote authentication dataset [1].

    [1] https://archive.ics.uci.edu/ml/datasets/banknote+authentication
    :return: banknote authentication dataset
    """
    import pandas as pd
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
    df = pd.read_csv(url, header=None)
    df.columns = ["var", "skew", "curt", "entropy", "class"]
    return df



def load_breast_cancer() -> pd.DataFrame:
    """
    Load breast cancer wisconsin (diagnostic) dataset [1].

    Classes: 0 = malignant, 1 = benign

    [1] W.N. Street, W.H. Wolberg and O.L. Mangasarian. Nuclear feature extraction for breast tumor diagnosis.
        IS&T/SPIE 1993 International Symposium on Electronic Imaging: Science and Technology, volume 1905,
        pages 861-870, San Jose, CA, 1993.
    """
    data, targets = datasets.load_breast_cancer(return_X_y=True, as_frame=True)
    data["diagnosis"] = targets
    return data

def load_digits(n_class:int=3) -> pd.DataFrame:
    """
    Load the digits dataset [1].

    n_class: int, The number of classes to return. Between 0 and 10. Default in sklearn: 10.

    [1] https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits
    """
    data, targets = datasets.load_digits(return_X_y=True, as_frame=True, n_class=n_class)
    data["digit"] = targets
    data = data.iloc[:-2, :] # we remove the last two lines to get a nbr of samples which
                             # allows an easy train-test split
    return data


DATASETS = {
    "Wine": (load_wine, "cultivator"),
    "Breast Cancer": (load_breast_cancer, "diagnosis"),
    "Digits": (lambda: load_digits(n_class=3), "digit"),
    "Pima Diabetes": (load_pima, "Outcome"),
    "Iris": (load_iris, "species"),
    "Banknotes": (load_banknote, "class")

}

DEFAULT_DS = "Wine"


# ------------------------- Plotting helper functions --------------------------


def plot_digit(data: pd.DataFrame, dataframe_index: int):
    """
    Creates a matplotlib plot for a sample digit
    """
    feature_names = data.columns[:-1]

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(
        data[feature_names].values[dataframe_index, :].reshape(8, 8),
        cmap="gray"
    )
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def plot_knn_k_range_plotly(X_train, y_train, X_val, y_val, k_range, plot_train: bool = True) -> go.Figure:
    k_vals = list(k_range)
    scores_val = []
    scores_train = []

    for k in k_vals:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        scores_val.append(knn.score(X_val, y_val))
        scores_train.append(knn.score(X_train, y_train))

    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            x=k_vals,
            y=scores_val,
            mode="markers",
            name="Validation",
            marker=dict(size=9, color='blue', opacity=1),
            hovertemplate="k=%{x}<br>Val acc=%{y:.3f}<extra></extra>",
        )
    )

    if plot_train:
        fig.add_trace(
            go.Scatter(
                x=k_vals,
                y=scores_train,
                mode="markers",
                name="Train",
                marker=dict(size=11, color='yellow', opacity=0.7),
                hovertemplate="k=%{x}<br>Train acc=%{y:.3f}<extra></extra>",
            )
        )

    fig.update_layout(
        title="k-NN: accuracy over k",
        xaxis_title="k",
        yaxis_title="Accuracy",
        xaxis=dict(tickmode="array", tickvals=k_vals),
        margin=dict(l=60, r=20, t=50, b=50),
    )

    return fig


def plot_rf_n_trees_plotly(X_train, y_train, X_val, y_val, n_trees_range, plot_train: bool = True) -> go.Figure:
    n_vals = list(n_trees_range)
    scores_val = []
    scores_train = []

    for n in n_vals:
        rf = RandomForestClassifier(n_estimators=n, random_state=42)
        rf.fit(X_train, y_train)
        scores_val.append(rf.score(X_val, y_val))
        scores_train.append(rf.score(X_train, y_train))

    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            x=n_vals,
            y=scores_val,
            mode="markers",
            name="Validation",
            marker=dict(size=9, color='blue', opacity=1),
            hovertemplate="#Trees=%{x}<br>Val acc=%{y:.3f}<extra></extra>",
        )
    )

    if plot_train:
        fig.add_trace(
            go.Scatter(
                x=n_vals,
                y=scores_train,
                mode="markers",
                name="Train",
                marker=dict(size=11, opacity=0.7, color='yellow'),
                hovertemplate="#Trees=%{x}<br>Train acc=%{y:.3f}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Random Forest: accuracy over number of trees",
        xaxis_title="Number of trees",
        yaxis_title="Accuracy",
        xaxis=dict(tickmode="array", tickvals=n_vals),
        margin=dict(l=60, r=20, t=50, b=50),
    )

    return fig


def plot_rf_max_depth_plotly(X_train, y_train, X_val, y_val, max_depth_range, plot_train: bool = True) -> go.Figure:
    d_vals = list(max_depth_range)
    scores_val = []
    scores_train = []

    for d in d_vals:
        rf = RandomForestClassifier(max_depth=d, random_state=42)
        rf.fit(X_train, y_train)
        scores_val.append(rf.score(X_val, y_val))
        scores_train.append(rf.score(X_train, y_train))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=d_vals,
            y=scores_val,
            mode="markers",
            name="Validation",
            marker=dict(size=9, color='blue', opacity=1),
            hovertemplate="max_depth=%{x}<br>Val acc=%{y:.3f}<extra></extra>",
        )
    )

    if plot_train:
        fig.add_trace(
            go.Scatter(
                x=d_vals,
                y=scores_train,
                mode="markers",
                name="Train",
                marker=dict(size=11, color='yellow', opacity=0.7),
                hovertemplate="max_depth=%{x}<br>Train acc=%{y:.3f}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Random Forest: accuracy over max_depth",
        xaxis_title="max_depth",
        yaxis_title="Accuracy",
        xaxis=dict(tickmode="array", tickvals=d_vals),
        margin=dict(l=60, r=20, t=50, b=50),
    )

    return fig


def plot_dt_max_depth_plotly(X_train, y_train, X_val, y_val, max_depth_range, plot_train: bool = True) -> go.Figure:
    d_vals = list(max_depth_range)
    scores_val = []
    scores_train = []

    for d in d_vals:
        dt = DecisionTreeClassifier(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        scores_val.append(dt.score(X_val, y_val))
        scores_train.append(dt.score(X_train, y_train))

    fig = go.Figure()



    fig.add_trace(
        go.Scatter(
            x=d_vals,
            y=scores_val,
            mode="markers",
            name="Validation",
            marker=dict(size=9, color='blue', opacity=1),
            hovertemplate="max_depth=%{x}<br>Val acc=%{y:.3f}<extra></extra>",
        )
    )

    if plot_train:
        fig.add_trace(
            go.Scatter(
                x=d_vals,
                y=scores_train,
                mode="markers",
                name="Train",
                marker=dict(size=11, color='yellow', opacity=0.7),
                hovertemplate="max_depth=%{x}<br>Train acc=%{y:.3f}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Decision Tree: accuracy over max_depth",
        xaxis_title="max_depth",
        yaxis_title="Accuracy",
        xaxis=dict(tickmode="array", tickvals=d_vals),
        margin=dict(l=60, r=20, t=50, b=50),
    )

    return fig


def plot_decision_boundaries_plotly(
    classifier,
    X: pd.DataFrame,
    y,
    feature_pair,
    plot_range_percentage: float = 0.05,
) -> go.Figure:
    """
    Plot decision boundary and training points for a single feature pair in 2D using Plotly.
    """
    # feature_pair can be ["feat1", "feat2"] or ("feat1", "feat2")
    if isinstance(feature_pair[0], str) and len(feature_pair) == 2:
        feature_names = list(feature_pair)
    else:
        raise ValueError("feature_pair must be a list/tuple of two feature names")

    X_mat = X[feature_names].values
    y_series = y if isinstance(y, pd.Series) else pd.Series(y, name="Class")
    classes_sorted = sorted(y_series.unique())

    # color scheme
    base_colors = px.colors.qualitative.Set3
    colors = base_colors[: len(classes_sorted)]
    color_map = {cls: colors[i] for i, cls in enumerate(classes_sorted)}

    # area with additional margin
    x_min, x_max = X_mat[:, 0].min(), X_mat[:, 0].max()
    y_min, y_max = X_mat[:, 1].min(), X_mat[:, 1].max()
    x_border = (x_max - x_min) * plot_range_percentage
    y_border = (y_max - y_min) * plot_range_percentage
    x_min, x_max = x_min - x_border, x_max + x_border
    y_min, y_max = y_min - y_border, y_max + y_border

    grid_res = 150
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_res),
        np.linspace(y_min, y_max, grid_res),
    )

    # fit classifier on this feature pair
    clf = clone(classifier)
    clf.fit(X_mat, y_series)
    pred = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # map classes to ints for discrete colors
    class_to_idx = {cls: i for i, cls in enumerate(classes_sorted)}
    z = np.vectorize(class_to_idx.get)(pred)

    if len(classes_sorted) > 1:
        colorscale = [
            (i / (len(classes_sorted) - 1), color_map[cls])
            for i, cls in enumerate(classes_sorted)
        ]
    else:
        colorscale = [(0.0, colors[0]), (1.0, colors[0])]

    fig = go.Figure()

    # Decision regions (Heatmap)
    fig.add_trace(
        go.Heatmap(
            x=np.linspace(x_min, x_max, grid_res),
            y=np.linspace(y_min, y_max, grid_res),
            z=z,
            colorscale=colorscale,
            showscale=False,
            opacity=0.5,
            hoverinfo="skip",
        )
    )

    # training samples per class
    for cls in classes_sorted:
        mask = y_series == cls
        fig.add_trace(
            go.Scatter(
                x=X_mat[mask, 0],
                y=X_mat[mask, 1],
                mode="markers",
                name=f"{y_series.name}: {cls}",
                marker=dict(
                    color=color_map[cls],
                    size=7,
                    line=dict(width=1, color="black"),
                ),
                hovertemplate=(
                    f"{feature_names[0]}=%{{x:.3f}}<br>"
                    f"{feature_names[1]}=%{{y:.3f}}<br>"
                    f"{y_series.name}={cls}<extra></extra>"
                ),
            )
        )

    fig.update_xaxes(title_text=feature_names[0], range=[x_min, x_max])
    fig.update_yaxes(title_text=feature_names[1], range=[y_min, y_max])

    fig.update_layout(
        title=f"Decision boundary ({feature_names[0]} vs. {feature_names[1]})",
        margin=dict(l=60, r=20, t=60, b=60),
        legend=dict(x=1.02, y=1.0),
        plot_bgcolor="white",
    )

    return fig



# ------------------------- Custom CSS & JS -----------------------------------

head = ui.head_content(
    ui.tags.link(
        rel="stylesheet",
        href="vendor/zephyr/bootstrap.min.css"
    ),
    ui.tags.style("""
    /* Remove border around nav tabs and sidebar panels */
    .nav-tabs {
        border: none !important;
        box-shadow: none !important;
    }
    /* Remove background highlight so it blends fully */
    .tab-content {
        background-color: transparent !important;
    }
    """),
    ui.tags.style("""
    /* Disabling absolute fill on plots */
    .shiny-plot-output.html-fill-item,
    .shiny-output-plot.html-fill-item {
    position: static !important;
    width: auto !important;
    max-width: 100%;
    display: inline-block !important;
    }

    /* A simple centering helper */
    .plot-center {
    display: flex;
    justify-content: center;
    }
    """),
    ui.tags.style("""
    .poly-plot-container {
        width: 100%;
        display: flex;
        justify-content: center;    /* center inner widget */
        align-items: center;
        margin-left: auto;          /* center the container inside the card */
        margin-right: auto;
    }
                  
    .poly-plot-container > #poly_fit_plot {
        width: 100%;
        max-width: 700px;           /* optional: cap size; remove if you want full width */
        aspect-ratio: 1 / 1;        /* keeps it square */
    }

    /* htmlwidget + inner Plotly-Div to 100% */
    #poly_fit_plot, 
    #poly_fit_plot > div {
        height: 100%;
        width: 100%;
    }
    """),
    ui.tags.style("""
    .cm-plot-container,
    .curve-plot-container,
    .feat-importance-plot-container {
        width: 100%;
        display: flex;
        justify-content: center;    /* center inner widget */
        align-items: center;
        margin-left: auto;          /* center the container inside the card */
        margin-right: auto;
    }

    /* ----- Aspect ratio + responsiveness ----- */

    /* Confusion matrix: square, responsive */
    .cm-plot-container > #cm {
        width: 100%;
        max-width: 700px;           /* optional: cap size */
        aspect-ratio: 1 / 1;        /* keeps it square */
    }

    /* ROC/PR curves: also square */
    .curve-plot-container > #curve_plot {
        width: 100%;
        max-width: 700px;           /* optional */
        aspect-ratio: 1 / 1;
    }

    /* Feature importance: wider than tall (e.g. 4:3) */
    .feat-importance-plot-container > #feat_importance {
        width: 100%;
        max-width: 800px;           /* optional */
        aspect-ratio: 4 / 3;
    }

    /* inner Plotly div fills the container */
    #cm, #cm > div,
    #curve_plot, #curve_plot > div,
    #feat_importance, #feat_importance > div {
        width: 100% !important;
        height: 100% !important;
    }

    /* PCA plot container: center like the others */
    .pca-plot-container {
        width: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
        margin-left: auto;
        margin-right: auto;
    }

    /* PCA widget: responsive square */
    .pca-plot-container > #pca_scatter {
        width: 100%;
        max-width: 700px;          /* optional cap */
        aspect-ratio: 1 / 1;       /* keep PC1 vs PC2 square */
    }

    /* inner Plotly div fills the container */
    #pca_scatter,
    #pca_scatter > div {
        width: 100% !important;
        height: 100% !important;
    }
    .pairplot-all-container,
    .pairplot-train-container,
    .pairplot-test-container {
        width: 100%;
        display: flex;
        justify-content: center;   /* center Plotly widget in container */
        align-items: center;
        margin-left: auto;         /* center container in card */
        margin-right: auto;
    }

    /* size, aspect ratio, not square but wider */
    .pairplot-all-container > #feature_pairplot_all_samples,
    .pairplot-train-container > #train_pairplot,
    .pairplot-test-container > #test_pairplot {
        width: 100%;
        max-width: 900px;          /* optional, otherwise full width */
        aspect-ratio: 4 / 3;
    }

    /* inner Plotly div fills the container */
    #feature_pairplot_all_samples, #feature_pairplot_all_samples > div,
    #train_pairplot, #train_pairplot > div,
    #test_pairplot, #test_pairplot > div {
        width: 100% !important;
        height: 100% !important;
    }

    .knn-k-range-container,
    .rf-ntrees-container,
    .rf-maxdepth-container,
    .dt-maxdepth-container,
    .decision-boundary-container {
        width: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
        margin-left: auto;
        margin-right: auto;
    }

    /* aspect ratios */
    .knn-k-range-container > #knn_k_range_plot,
    .rf-ntrees-container > #rf_n_trees_plot,
    .rf-maxdepth-container > #rf_max_depth_plot,
    .dt-maxdepth-container > #dt_max_depth_plot {
        width: 100%;
        max-width: 800px;
        aspect-ratio: 16 / 9;
    }

    .decision-boundary-container > #decision_boundary {
        width: 100%;
        max-width: 700px;
        aspect-ratio: 1 / 1;
    }

    /* Plotly-Divs fill their container */
    #knn_k_sweep, #knn_k_sweep > div,
    #rf_trees_sweep, #rf_trees_sweep > div,
    #rf_depth_sweep, #rf_depth_sweep > div,
    #dt_depth_sweep, #dt_depth_sweep > div,
    #decision_boundary, #decision_boundary > div {
        width: 100% !important;
        height: 100% !important;
    }

    """),

    ui.tags.link(
        rel="stylesheet",
        href="https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/katex.min.css",
        crossorigin="anonymous"
    ),
    ui.tags.script(src="https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/katex.min.js"),
    ui.tags.script(src="https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/contrib/auto-render.min.js"),
    ui.tags.script("""
    document.addEventListener('DOMContentLoaded', function() {
        renderMathInElement(document.body);
    });
    """),
)

# ------------------------- UI Layout -----------------------------------------

app_ui = ui.page_fluid(
    head,
    ui.h2("Hands-on AI I - Unit 4: Supervised ML Basics", class_="text-center mb-3"),
    ui.navset_tab(
        ui.nav_panel("Function Fitting",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_select("function", "Function", ["Noisy sine", "Mystery function"], selected="Noisy sine"),
                    ui.input_slider("n_samples", "Samples", 10, 200, 40),
                    ui.input_slider("noise", "Noise σ", 0.0, 0.6, 0.2, step=0.01),
                    ui.input_numeric("seed", "Random seed", value=123, min=0, max=999, step=1),
                    ui.input_checkbox("true_curve", "Show true curve", False),
                    ui.input_slider("degrees", "Polynomial degree(s)", 1, 50, value=1, step=1),
                    ui.input_action_button("fit", "Fit polynomial", class_="btn btn-primary"),
                ),
                ui.card(
                    ui.card_header("Curve fitting & MSE"),
                    ui.div(
                        output_widget("poly_fit_plot", width="100%", height="100%"),
                        class_="poly-plot-container"
                    ),
                ),
                ui.card(
                    ui.card_header("Fitting info"),
                    ui.output_text_verbatim("poly_mse_info"),
                )
            )
        ),
        ui.nav_panel("Data Settings & PCA",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_select("dataset", "Dataset", list(DATASETS.keys()), selected=DEFAULT_DS),
                    ui.input_numeric("data_seed", "Seed", 42, min=0, max=9999, step=1),
                    ui.input_checkbox("standardize", "Standardize features (zero-mean, unit-var)", True),
                    ui.hr(),
                    ui.input_slider("split", "Train % (of all data)", min=50, max=90, value=70, step=5),
                    ui.input_slider("val_split", "Validation % (of all data)", min=5, max=40, value=15, step=5),
                    ui.output_text_verbatim("split_overview"),
                    ui.hr(),
                    ui.input_checkbox("use_all_features", "Use all features in train/val/test split", True),
                    ui.panel_conditional(
                        "input.use_all_features == false",
                        ui.input_selectize("feature_select", "Select features for train/val/test split", choices=[], selected=[], multiple=True, width="100%"),
                    ),
                    ui.hr(),
                    ui.input_action_button("load", "Load dataset", class_="btn btn-primary"),
                ),
                ui.navset_card_pill(
                    ui.nav_panel("Dataset",
                        ui.output_data_frame("data_table")
                    ),
                    ui.nav_panel("Summary Statistics",
                        ui.output_text_verbatim("data_summary")
                    ),
                    ui.nav_panel("PCA",
                        ui.card(
                            ui.div(
                                output_widget("pca_scatter", width="100%", height="100%"),
                                class_="pca-plot-container"
                            ),
                        ),
                        ui.tags.div(style="margin-top: 20px;"),
                        ui.output_text_verbatim("pca_explained"),
                        ui.tags.div(style="margin-top: 10px;"),
                        ui.output_text_verbatim("pca_loadings_top_features")
                    ),
                    ui.nav_panel("Feature Pairplots",
                        ui.row(
                            ui.column(6,
                                ui.input_selectize(
                                    "pairplot_features_all_samples",
                                    "Select features for pairplot",
                                    choices=[], selected=[], multiple=True, width="100%"
                                )
                            ),
                            ui.column(6,
                                ui.div(
                                    ui.input_action_button("plot_pairplots_all_samples", "Plot", class_="btn btn-primary"),
                                    style="margin-top: 30px;",
                                ),
                            ),
                        ),
                        ui.card(
                            ui.div(
                                output_widget("feature_pairplot_all_samples", width="100%", height="100%"),
                                class_="pairplot-all-container",
                            ),
                        ),
                    ),
                    ui.nav_panel("Train/Test Set Pairplots",
                        ui.row(
                            ui.column(6,
                                ui.input_selectize(
                                    "pairplot_features",
                                    "Select features for pairplot",
                                    choices=[], selected=[], multiple=True, width="100%"
                                )
                            ),
                            ui.column(6,
                                ui.div(
                                    ui.input_action_button("plot_pairplots", "Plot", class_="btn btn-primary"),
                                    style="margin-top: 30px;",
                                ),
                            ),
                        ),
                        ui.card(
                            ui.card_header("Training set pairplot"),
                            ui.div(
                                output_widget("train_pairplot", width="100%", height="100%"),
                                class_="pairplot-train-container",
                            ),
                        ),
                        ui.tags.div(style="margin-top: 20px;"),
                        ui.card(
                            ui.card_header("Test set pairplot"),
                            ui.div(
                                output_widget("test_pairplot", width="100%", height="100%"),
                                class_="pairplot-test-container",
                            ),
                        ),
                    ),
                title="Data Exploration"),
                ui.card(
                    ui.card_header("Dataset Info"),
                    ui.panel_conditional("input.dataset == 'Wine'",
                        ui.p("The Wine dataset consists of samples of wines from three different cultivators, with several features gained from chemical analysis. The goal is to classify the cultivator based on these features.")
                    ),
                    ui.panel_conditional("input.dataset == 'Breast Cancer'",
                        ui.p("The Breast Cancer Wisconsin dataset contains samples of breast tissue, with features computed from digitized images of fine needle aspirate (FNA) of breast masses. The task is to classify the tissue as malignant or benign.")
                    ),
                    ui.panel_conditional("input.dataset == 'Digits'",
                        ui.p("The Digits dataset includes samples of handwritten digits (0-9) represented as 8x8 pixel images. In this app, we only use the first three classes (digits 0, 1, and 2) for classification."),
                    ),
                    ui.panel_conditional("input.dataset == 'Pima Diabetes'",
                        ui.p("The Pima Indians Diabetes dataset consists of samples with medical predictor features. The goal is to predict whether a patient has diabetes (Outcome) based on these features.")
                    ),
                    ui.panel_conditional("input.dataset == 'Iris'",
                        ui.p("The Iris dataset contains samples of iris flowers from three different species, with four features: sepal length, sepal width, petal length, and petal width. The task is to classify the species based on these features.")
                    ),
                    ui.panel_conditional("input.dataset == 'Banknotes'",
                        ui.p("The Banknote Authentication dataset includes samples of genuine and forged banknotes, with features extracted from images of the banknotes. The goal is to classify the banknotes as genuine or forged.")
                    ),
                    fill=False
                ),
                ui.panel_conditional("input.dataset == 'Digits' && input.load > 0",
                    ui.card(
                        ui.card_header("Sample Digits"),
                        ui.input_numeric("digit_sample_index", "Select a sample index", 0, min=0, max=534, step=1),
                        ui.output_plot("digit_samples", height="320px")
                    )
                )
            )
        ),
        ui.nav_panel("Classification Models",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h4("Classifier Settings"),
                    ui.input_select("classifier", "Classifier", ["k-NN", "Decision Tree", "Random Forest"], selected="k-NN"),
                    ui.input_numeric("classifier_seed", "Seed", 42, min=0, max=9999, step=1),
                    ui.panel_conditional(
                        "input.classifier == 'k-NN'",
                        ui.input_slider("knn_k", "Number of neighbors k", 1, 30, 3, step=1),
                    ),
                    ui.panel_conditional(
                        "input.classifier == 'Decision Tree'",
                        ui.input_slider("dt_depth", "Max tree depth (0 = unlimited)", 0, 20, 5, step=1),
                    ),
                    ui.panel_conditional(
                        "input.classifier == 'Random Forest'",
                        ui.input_slider("rf_trees", "Number of trees", 10, 100, 50, step=5),
                        ui.input_slider("rf_depth", "Max tree depth (0 = unlimited)", 0, 20, 5, step=1),
                    ),
                    ui.hr(),
                    ui.h4("Data"),
                    ui.output_text_verbatim("selected_data"),
                    ui.hr(),
                    ui.input_action_button("apply_classifier", "Apply classifier", class_="btn btn-primary"),

                ),
                ui.navset_card_pill(
                    ui.nav_panel("Decision Boundary",
                        ui.column(
                            5,
                            ui.input_selectize(
                                "db_features",
                                "Select features for decision boundary (exactly 2)",
                                choices=[], selected=[], multiple=True, width="100%",
                            ),
                        ),
                        ui.card(
                            ui.div(
                                output_widget("decision_boundary", width="100%", height="100%"),
                                class_="decision-boundary-container",
                            ),
                        ),
                    ),
                    ui.nav_panel("Confusion Matrix",
                        ui.row(
                            ui.column(6,
                                ui.input_radio_buttons(
                                    "cm_normalize", "Normalize",
                                    choices=["Off", "Row %", "Column %", "Overall %"],
                                    selected="Off",
                                    inline=True
                                ),
                            ),
                        ),
                        ui.card(
                            ui.div(
                                output_widget("cm", width="100%", height="100%"),
                                class_="cm-plot-container"
                            ),
                        )
                    ),
                    ui.nav_panel("Classification Report",
                        ui.output_text_verbatim("class_report"),
                        ui.tags.div(style="margin-top: 20px;"),
                        ui.card(
                            ui.card_header("Accuracy"),
                            ui.p("The accuracy is the overall fraction of correctly classified samples."),
                        ),
                        ui.tags.div(style="margin-top: 10px;"),
                        ui.card(
                            ui.card_header("Precision"),
                            ui.p("The precision is the fraction of true positive predictions among all positive predictions, i.e., it measures how many of the predicted positives are actually positive. The formal definition is given by \(\\mathrm{Precision} = \\frac{TP}{TP + FP}\), where \(TP\) are the true positives and \(FP\) the false positives."),
                        ),
                        ui.tags.div(style="margin-top: 10px;"),
                        ui.card(
                            ui.card_header("Recall"),
                            ui.p("The recall is the fraction of true positive predictions among all actual positives, i.e., it measures how many of the actual positives were correctly identified. The formal definition is given by \(\\mathrm{Recall} = \\frac{TP}{TP + FN}\), where \(TP\) are the true positives and \(FN\) the false negatives."),
                        ),
                        ui.tags.div(style="margin-top: 10px;"),
                        ui.card(
                            ui.card_header("F1 Score"),
                            ui.p("The F1 score is the harmonic mean of precision and recall and thus balances the trade-off between precision and recall, i.e., it is high only if both precision and recall are high. We cannot maximize both, but we want to find the best trade-off (also see the precision-recall curve). The formal definition is given by \(F1 = 2 \\cdot \\frac{\\mathrm{Precision} \\cdot \\mathrm{Recall}}{\\mathrm{Precision} + \\mathrm{Recall}}\\)."),
                        ),
                        ui.tags.div(style="margin-top: 10px;"),
                        ui.card(
                            ui.card_header("Weighted vs. Macro Average"),
                            ui.p("The weighted average aggregates the metrics for all classes, weighted by the number of samples in each class. This is useful when dealing with imbalanced datasets, as it gives more importance to the performance on the majority classes. The macro average computes the metrics for each class independently and then takes the average, treating all classes equally regardless of their size."),
                        ),
                        ui.tags.div(style="margin-top: 10px;"),
                        ui.card(
                            ui.card_header("Support"),
                            ui.p("The support is the number of actual occurrences of each class in the dataset."),
                        ),
                    ),
                    ui.nav_panel("Hyperparameter Optimization",
                    ui.tags.div(style="margin-top: 20px;"),
                    ui.panel_conditional(
                        "input.classifier == 'k-NN'",
                        ui.h4("k-NN: Accuracy vs. k"),
                        ui.input_slider("knn_k_range", "Select k range", 2, 30, [2, 15], step=1),
                        ui.tags.div(style="margin-top: 20px;"),
                        ui.card(
                            ui.div(
                                output_widget("knn_k_sweep", width="100%", height="100%"),
                                class_="plotly-plot-container",
                                style="height: 500px;",
                            ),
                        ),
                    ),
                    ui.panel_conditional(
                        "input.classifier == 'Random Forest'",
                        ui.h4("Random Forest: Accuracy vs. Number of Trees"),
                        ui.input_slider("rf_trees_range", "Select number of trees", 10, 100, [10, 30], step=10),
                        ui.tags.div(style="margin-top: 20px;"),
                        ui.card(
                            ui.div(
                                output_widget("rf_trees_sweep", width="100%", height="100%"),
                                class_="plotly-plot-container",
                                style="height: 500px;",
                            ),
                        ),
                    ),
                    ui.panel_conditional(
                        "input.classifier == 'Random Forest'",
                        ui.tags.div(style="margin-top: 20px;"),
                        ui.h4("Random Forest: Accuracy vs. Max Depth"),
                        ui.input_slider("rf_depth_range", "Select max depth", 1, 20, [1, 5], step=1),
                        ui.tags.div(style="margin-top: 20px;"),
                        ui.card(
                            ui.div(
                                output_widget("rf_depth_sweep", width="100%", height="100%"),
                                class_="plotly-plot-container",
                                style="height: 500px;",
                            ),
                        ),
                    ),
                    ui.panel_conditional(
                        "input.classifier == 'Decision Tree'",
                        ui.h4("Decision Tree: Accuracy vs. Max Depth"),
                        ui.input_slider("dt_depth_range", "Select max depth", 1, 20, [1, 5], step=1),
                        ui.tags.div(style="margin-top: 20px;"),
                        ui.card(
                            ui.div(
                                output_widget("dt_depth_sweep", width="100%", height="100%"),
                                class_="plotly-plot-container",
                                style="height: 500px;",
                            ),
                        ),
                    ),
                ),
                    ui.nav_panel("Model Insights",
                        ui.h4("Feature Importance Settings"),
                        ui.row(
                            ui.column(4,
                                ui.div(
                                    ui.input_radio_buttons(
                                        "fi_method", "Importance method",
                                        choices=["Model-based", "Permutation"],
                                        selected="Model-based", inline=True
                                    ),
                                style="margin-top: 18px;"),
                            ),
                            ui.column(4,
                                ui.div(
                                    ui.input_checkbox("fi_normalize", "Normalize importances to sum = 1", True),
                                    style="margin-top: 50px;",
                                ),
                            ),
                            ui.column(4,
                                ui.input_slider("fi_topn", "Show top n features", min=1, max=20, value=10, step=1),
                            ),
                        ),
                        ui.hr(),
                        ui.panel_conditional(
                            "input.fi_method == 'Permutation'",
                            ui.row(
                                ui.column(4,
                                    ui.input_slider("fi_repeats", "Permutation repeats", min=1, max=50, value=10, step=1),              
                                ),
                                ui.column(4,
                                    ui.input_select(
                                        "fi_scoring", "Scoring",
                                        choices=["accuracy", "f1_macro", "f1_weighted"],
                                        selected="accuracy"
                                    ),
                                ),
                                style="align-items: center; margin-top: 10px;",
                            ),
                            ui.hr(),
                        ),
                        ui.card(
                            ui.div(
                                output_widget("feat_importance", width="100%", height="100%"),
                                class_="feat-importance-plot-container"
                            ),
                        ),
                        ui.tags.div(style="margin-top: 20px;"),
                        ui.card(
                            ui.card_header("Model-based vs. Permutation Feature Importances"),
                            ui.p("Model-based feature importances (e.g., Gini importance in Random Forests) are derived from how much a feature contributes to reducing impurity across all splits in the trees. However, this does not directly measure impact on predictions and can be biased toward:"),
                            ui.tags.ul(
                                ui.tags.li("Features with many possible split points (continuous or high-cardinality features)"),
                                ui.tags.li("Features with high variance"),
                                ui.tags.li("One-hot encoded categorical features")
                            ),
                            ui.p("Permutation feature importance instead measures how much the model’s performance (e.g., accuracy or F1) deteriorates when the values of a single feature are randomly permuted. This directly reflects how important the feature is for the model’s actual predictions and works for any model. However, if features are strongly correlated, permutation importance may underestimate their importance, because the model can rely on the correlated feature as a substitute."),
                        )
                    ),
                    ui.nav_panel("ROC / PR Curves",
                        ui.row(
                            ui.column(6, ui.input_select("curve_metric", "Metric", ["ROC", "PR"], selected="ROC")),
                            ui.panel_conditional(
                                "input.dataset == 'Wine' || input.dataset == 'Digits' || input.dataset == 'Iris'",
                                ui.column(6,
                                    ui.input_select("curve_class", "Class (multiclass one-vs-rest)", choices=[], selected=None),
                                )
                            )
                        ),
                        ui.card(
                            ui.div(
                                output_widget("curve_plot", width="100%", height="100%"),
                                class_="curve-plot-container"
                            ),
                        )
                    ),
                    ui.nav_panel("Error Analysis",
                        ui.h4("Misclassified Samples"),
                        ui.p("Shows all misclassified samples from the test set. Sorted by predicted probability of the (wrong) predicted class."),
                        ui.hr(),
                        ui.output_data_frame("errors_table"),
                    ),
                    title="Model Evaluation"
                )
            )
        )
    )
)



def server(input, output, session):

    model = reactive.Value(None)
    Xtr = reactive.Value(None)
    Xte = reactive.Value(None)
    ytr = reactive.Value(None)
    yte = reactive.Value(None)
    Xva = reactive.Value(None)
    yva = reactive.Value(None)
    data_rev = reactive.Value(0)   # changes when data changes
    model_rev = reactive.Value(-1) # version, against which the model was trained



    def _tt_split(*args, **kwargs):
        try:
            return train_test_split(*args, **kwargs, stratify=kwargs.get("y_full", None))
        except ValueError:
            # fallback without stratify
            k = dict(kwargs); k.pop("stratify", None)
            return train_test_split(*args, **k)

    @reactive.calc
    def compute_splits():
        if input.load() == 0:
            return None

        df = df_loaded()
        tgt = DATASETS[input.dataset()][1]

        feats = (list(input.feature_select())
                if not input.use_all_features()
                else [c for c in df.select_dtypes(include=[np.number]).columns if c != tgt])

        if len(feats) < 2 or tgt not in df.columns:
            model.set(None)
            Xtr.set(None); Xva.set(None); Xte.set(None)
            ytr.set(None); yva.set(None); yte.set(None)
            return None

        # compute splits
        X_full = df[feats].copy()
        y_full = df[tgt].copy()

        train_pct = float(input.split()) / 100.0
        val_pct   = float(input.val_split()) / 100.0
        test_pct  = max(0.0, 1.0 - train_pct - val_pct)

        # split off test set (if test_pct > 0)
        if test_pct > 0:
            try:
                X_trainval, X_test, y_trainval, y_test = train_test_split(
                    X_full, y_full,
                    test_size=test_pct,
                    random_state=int(input.data_seed()),
                    stratify=y_full
                )
            except ValueError:
                # fallback without stratify
                X_trainval, X_test, y_trainval, y_test = train_test_split(
                    X_full, y_full,
                    test_size=test_pct,
                    random_state=int(input.data_seed()),
                )
        else:
            X_trainval, y_trainval = X_full, y_full
            X_test = None
            y_test = None

        # train/val in the rest (only if val_pct > 0)
        if val_pct > 0:
            rel_val = val_pct / (train_pct + val_pct)
            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_trainval, y_trainval,
                    test_size=rel_val,
                    random_state=int(input.data_seed()),
                    stratify=y_trainval
                )
            except ValueError:
                # Fallback ohne Stratify
                X_train, X_val, y_train, y_val = train_test_split(
                    X_trainval, y_trainval,
                    test_size=rel_val,
                    random_state=int(input.data_seed()),
                )
        else:
            X_train, y_train = X_trainval, y_trainval
            X_val = None
            y_val = None

        # optional scaling: fit on train, then transform val/test
        if input.standardize():
            scaler = StandardScaler().fit(X_train)
            X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
            if X_val is not None:
                X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
            if X_test is not None:
                X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

        # write to reactive.Values
        Xtr.set(X_train); ytr.set(y_train)
        Xva.set(X_val);   yva.set(y_val)
        Xte.set(X_test);  yte.set(y_test)

        return {
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val,     "y_val": y_val,
            "X_test": X_test,   "y_test": y_test,
            "features": feats, "target": tgt,
            "pct": (train_pct, val_pct, test_pct),
        }

    @reactive.calc
    def split_percentages():
        train_pct = float(input.split())
        val_pct   = float(input.val_split())
        test_pct  = max(0, 100 - train_pct - val_pct)
        return train_pct, val_pct, test_pct

    @reactive.effect
    @reactive.event(
        input.load, input.dataset, input.split, input.val_split,
        input.standardize, input.use_all_features, input.feature_select
    )
    def _ensure_splits_fresh():
        _ = compute_splits()

    @reactive.effect
    def _enforce_min_test_on_start():
        max_val = 100 - input.split() - 5
        if input.val_split() > max_val:
            ui.update_slider("val_split", max=max_val, value=max_val)

    @reactive.effect
    @reactive.event(input.load, input.dataset)
    def _update_digit_index_max():
        if input.dataset() == "Digits":
            df = df_loaded()
            ui.update_numeric("digit_sample_index", max=max(0, len(df) - 1))


    @reactive.calc
    def df_loaded() -> pd.DataFrame:
        fn, tgt = DATASETS[input.dataset()]
        df = fn()
        # reorder columns: numeric first, then others, target last (if exists)
        num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        other_cols = [c for c in df.columns if c not in num_cols and c != tgt]
        ordered = num_cols + other_cols + ([tgt] if tgt in df.columns and tgt not in num_cols else [])
        ordered = [c for i, c in enumerate(ordered) if c in df.columns and c not in ordered[:i]]
        return df[ordered] if ordered else df
    


    # ------------ Function fitting  -------------------

    def _gen_noisy_sine(n: int, sigma: float, seed: int):
        rng = np.random.default_rng(seed)
        x = np.sort(rng.uniform(0, 1, size=n))
        y_true = np.sin(2*np.pi*x)
        y = y_true + rng.normal(0, sigma, size=n)
        return x, y, y_true
    
    def _gen_mystery_function(n: int, sigma: float, seed: int):
        rng = np.random.default_rng(seed)
        x = np.sort(rng.uniform(0, 1, size=n))
        y_true = 0.1*np.exp(2*(x+0.5)**2)*(np.sin(-1.5 * np.pi * x) - np.cos(2.3 * np.pi * x))
        y = y_true + rng.normal(0, sigma, size=n)
        return x, y, y_true
    
    def _gen_data(n: int, sigma: float, seed: int, func: str):
        if func == "Noisy sine":
            return _gen_noisy_sine(n, sigma, seed)
        elif func == "Mystery function":
            return _gen_mystery_function(n, sigma, seed)
        else:
            raise ValueError(f"Unknown function: {func}")


    @render_plotly
    @reactive.event(
        input.fit,
        input.n_samples,
        input.noise,
        input.seed,
        input.function,
        input.true_curve,
        input.degrees,
    )
    def poly_fit_plot():
        if input.fit() == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="Click 'Fit polynomial' to see fitting results",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=14),
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            fig.update_layout(
                autosize=True,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            return fig

        x, y, _ = _gen_data(
            input.n_samples(),
            input.noise(),
            input.seed(),
            input.function(),
        )
        xs = np.linspace(0, 1, 400)

        fig = go.Figure()

        if input.true_curve():
            if input.function() == "Noisy sine":
                ys_true = np.sin(2 * np.pi * xs)
            elif input.function() == "Mystery function":
                ys_true = 0.1 * np.exp(2 * (xs + 0.5) ** 2) * (
                    np.sin(-1.5 * np.pi * xs) - np.cos(2.3 * np.pi * xs)
                )
            else:
                ys_true = None

            if ys_true is not None:
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys_true,
                        mode="lines",
                        name="True curve",
                        line=dict(width=2),
                    )
                )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name="Samples",
                marker=dict(size=7, line=dict(width=1, color="black")),
            )
        )

        d = input.degrees()
        model_pipe = Pipeline(
            [
                ("poly", PolynomialFeatures(degree=d, include_bias=False)),
                ("lin", LinearRegression()),
            ]
        )
        model_pipe.fit(x.reshape(-1, 1), y)
        y_pred = model_pipe.predict(xs.reshape(-1, 1))

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=y_pred,
                mode="lines",
                name=f"deg {d}",
                line=dict(width=2),
            )
        )

        fig.update_layout(
            autosize=True,
            margin=dict(l=60, r=20, t=40, b=60),
            legend=dict(x=0.01, y=0.99),
        )
        fig.update_xaxes(
            title_text="x", 
            range=[0, 1]
        )

        fig.update_yaxes(
            title_text="y",
            range=[-3.2, 3.2],
        )

        return fig

    @render.text
    @reactive.event(input.fit, input.n_samples, input.noise, input.seed, input.function, input.degrees)
    def poly_mse_info():
        if input.fit() == 0:
            return "Click 'Fit polynomial' to see fitting results"
        x, y, _ = _gen_data(input.n_samples(), input.noise(), input.seed(), input.function())
        d = input.degrees()
        model_pipe = Pipeline([
            ("poly", PolynomialFeatures(degree=d, include_bias=False)),
            ("lin", LinearRegression()),
        ])
        model_pipe.fit(x.reshape(-1,1), y)
        y_hat_train = model_pipe.predict(x.reshape(-1,1))
        return f"Polynomial degree: {d}\nMSE (fitted): {round(float(mean_squared_error(y, y_hat_train)), 4)}\nCoefficients: {np.round(model_pipe.named_steps['lin'].coef_, 4)}"
    
    # ------------ Data settings & PCA ----------------------------------------

    @reactive.effect
    def update_val_slider():
        max_val = 100 - input.split() - 5  # Ensure at least 5% for test set
        if input.val_split() > max_val:
            ui.update_slider("val_split", max=max_val, value=max_val)
        else:
            ui.update_slider("val_split", max=max_val)

        

    @render.data_frame
    @reactive.event(input.load, input.dataset, input.use_all_features, input.feature_select)
    def data_table():
        if input.load() == 0:
            return pd.DataFrame({"Info": ["Click 'Load dataset' to display data"]})
        df = df_loaded()
        if input.use_all_features():
            return df
        selected = list(input.feature_select())
        tgt = DATASETS[input.dataset()][1]
        if tgt in df.columns and tgt not in selected:
            selected.append(tgt)
        if not selected:
            return pd.DataFrame({"Info": ["No features selected"]})
        df = df[[c for c in selected if c in df.columns]]
        return df
    
    @reactive.effect
    def update_feature_select_choices():
        df = df_loaded()
        tgt = DATASETS[input.dataset()][1]
        feats = [c for c in df.select_dtypes(include=[np.number]).columns if c != tgt]
        ui.update_selectize("feature_select", choices=feats, selected=feats if feats else [])
    
    @reactive.effect
    @reactive.event(input.load, input.dataset)
    def update_pairplot_features():
        if input.load() == 0:
            return
        df = df_loaded()
        tgt = DATASETS[input.dataset()][1]
        feats = list(input.feature_select()) if not input.use_all_features() else [c for c in df.select_dtypes(include=[np.number]).columns if c != tgt]
        ui.update_selectize("pairplot_features", choices=feats)

    @reactive.effect
    @reactive.event(input.load, input.dataset)
    def update_pairplot_features_all_samples():
        if input.load() == 0:
            return
        df = df_loaded()
        tgt = DATASETS[input.dataset()][1]
        feats = list(input.feature_select()) if not input.use_all_features() else [c for c in df.select_dtypes(include=[np.number]).columns if c != tgt]
        ui.update_selectize("pairplot_features_all_samples", choices=feats)

    @render_plotly
    @reactive.event(input.load, input.standardize, input.dataset)
    def pca_scatter():

        if input.load() == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="Click 'Load dataset' to display PCA",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=14),
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))
            return fig

        # Prepare data
        df = df_loaded()
        tgt = DATASETS[input.dataset()][1]
        feats = [c for c in df.select_dtypes(include=[np.number]).columns if c != tgt]

        if len(feats) < 2:
            fig = go.Figure()
            fig.add_annotation(
                text="Need at least two numeric features for PCA",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=14),
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))
            return fig

        X = df[feats].copy()
        y = df[tgt]

        if input.standardize():
            X[:] = StandardScaler().fit_transform(X)

        pca = PCA(n_components=2)
        pcs = pca.fit_transform(X)

        proj = pd.DataFrame(
            {
                "PC1": pcs[:, 0],
                "PC2": pcs[:, 1],
                tgt: y.values,
            }
        )

        # Build interactive scatter with one trace per class
        fig = go.Figure()
        for cls in sorted(proj[tgt].unique()):
            mask = proj[tgt] == cls
            fig.add_trace(
                go.Scatter(
                    x=proj.loc[mask, "PC1"],
                    y=proj.loc[mask, "PC2"],
                    mode="markers",
                    name=str(cls),
                    marker=dict(size=8, line=dict(width=1, color="black")),
                    hovertemplate=(
                        f"{tgt}={cls}<br>"
                        "PC1=%{x:.3f}<br>"
                        "PC2=%{y:.3f}<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            title=f"PCA projection (first 2 PCs) of {input.dataset()} (full feature set)",
            xaxis_title="PC 1",
            yaxis_title="PC 2",
            legend=dict(x=0.01, y=0.99),
            margin=dict(l=60, r=20, t=60, b=60),
            autosize=True,
        )

        # Preserve equal scaling for PC1 / PC2
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

        return fig

    def _plotly_pairplot(plot_df, features, target, title: str):
        """
        Plotly equivalent of seaborn.pairplot(diag_kind="hist", hue=target)
        with improved spacing to prevent overlapping axis labels.
        """
        n = len(features)
        fig = make_subplots(
            rows=n,
            cols=n,
            shared_xaxes=False,
            shared_yaxes=False,
            horizontal_spacing=0.06,
            vertical_spacing=0.06,
        )

        classes = list(plot_df[target].unique())
        colors = px.colors.qualitative.Plotly
        color_map = {cls: colors[i % len(colors)] for i, cls in enumerate(classes)}

        show_legend_done = False

        for row_i, fy in enumerate(features):
            for col_j, fx in enumerate(features):
                row = row_i + 1
                col = col_j + 1

                if row_i == col_j:
                    # Diagonal: histogram
                    for k, cls in enumerate(classes):
                        mask = plot_df[target] == cls
                        fig.add_trace(
                            go.Histogram(
                                x=plot_df.loc[mask, fx],
                                name=str(cls),
                                marker=dict(color=color_map[cls]),
                                opacity=0.55,
                                showlegend=(not show_legend_done and row == 1 and col == 1),
                            ),
                            row=row,
                            col=col,
                        )
                    show_legend_done = True
                    # axis title on diagonal
                    fig.update_xaxes(title_text=fx, title_font=dict(size=10), row=row, col=col)

                else:
                    # Off-diagonal: scatter
                    for cls in classes:
                        mask = plot_df[target] == cls
                        fig.add_trace(
                            go.Scatter(
                                x=plot_df.loc[mask, fx],
                                y=plot_df.loc[mask, fy],
                                mode="markers",
                                name=str(cls),
                                marker=dict(
                                    color=color_map[cls],
                                    size=5,
                                    line=dict(width=0.4, color="black"),
                                ),
                                showlegend=False,
                                hovertemplate=(
                                    f"{target}={cls}<br>"
                                    f"{fx}=%{{x:.3f}}<br>"
                                    f"{fy}=%{{y:.3f}}<extra></extra>"
                                ),
                            ),
                            row=row,
                            col=col,
                        )

                    # put axis labels only at the edges
                    if row == n:
                        fig.update_xaxes(title_text=fx, title_font=dict(size=10), row=row, col=col)
                    if col == 1:
                        fig.update_yaxes(title_text=fy, title_font=dict(size=10), row=row, col=col)

        fig.update_layout(
            title=title,
            barmode="overlay",
            autosize=True,
            height=None,
            width=None,
            margin=dict(l=80, r=20, t=60, b=60),  # increased left margin
            legend=dict(x=1.02, y=1.0),
            font=dict(size=10),                   # global smaller text
        )

        return fig



    
    @render_plotly
    @reactive.event(input.plot_pairplots_all_samples)
    def feature_pairplot_all_samples():
        if input.load() == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="Click 'Load dataset' to display pairplot",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=14),
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            return fig

        df = df_loaded()
        tgt = DATASETS[input.dataset()][1]
        selected = list(input.pairplot_features_all_samples())
        numeric_cols = set(df.select_dtypes(include=[np.number]).columns)
        selected = [f for f in selected if f in numeric_cols]

        if len(selected) < 2:
            fig = go.Figure()
            fig.add_annotation(
                text="Select at least two numeric features for pairplot",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=14),
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            return fig

        if tgt not in df.columns:
            fig = go.Figure()
            fig.add_annotation(
                text="Dataset has no target column for coloring",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=14),
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            return fig

        plot_df = df[selected].copy()
        plot_df[tgt] = df[tgt].values

        fig = _plotly_pairplot(
            plot_df,
            features=selected,
            target=tgt,
            title="Feature pairplot (full dataset)",
        )
        return fig



    @render_plotly
    @reactive.event(input.plot_pairplots)
    def train_pairplot():
        _ = compute_splits()
        X_train, y_train = Xtr(), ytr()
        if X_train is None or y_train is None:
            fig = go.Figure()
            fig.add_annotation(
                text="Train/Val/Test not yet computed.",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=14),
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            return fig

        df = df_loaded()
        tgt = DATASETS[input.dataset()][1]
        selected = list(input.pairplot_features())
        numeric_cols = set(df.select_dtypes(include=[np.number]).columns)
        selected = [f for f in selected if f in numeric_cols]

        if len(selected) < 2:
            fig = go.Figure()
            fig.add_annotation(
                text="Choose at least 2 features",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=14),
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            return fig

        plot_df = X_train[selected].copy()
        plot_df[tgt] = pd.Series(y_train).values

        train_pct, _, _ = split_percentages()
        fig = _plotly_pairplot(
            plot_df,
            features=selected,
            target=tgt,
            title=f"Train set pairplot ({train_pct:.1f}% of the data)",
        )
        return fig




    @render_plotly
    @reactive.event(input.plot_pairplots)
    def test_pairplot():
        _ = compute_splits()
        X_test, y_test = Xte(), yte()
        if X_test is None or y_test is None:
            fig = go.Figure()
            fig.add_annotation(
                text="Train/Val/Test not yet computed.",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=14),
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            return fig

        df = df_loaded()
        tgt = DATASETS[input.dataset()][1]
        selected = list(input.pairplot_features())
        numeric_cols = set(df.select_dtypes(include=[np.number]).columns)
        selected = [f for f in selected if f in numeric_cols]

        if len(selected) < 2:
            fig = go.Figure()
            fig.add_annotation(
                text="Select at least 2 features",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=14),
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            return fig

        plot_df = X_test[selected].copy()
        plot_df[tgt] = pd.Series(y_test).values

        _, _, test_pct = split_percentages()
        fig = _plotly_pairplot(
            plot_df,
            features=selected,
            target=tgt,
            title=f"Test set pairplot ({test_pct:.1f}% of the data)",
        )
        return fig


    @render.plot
    @reactive.event(input.load, input.dataset, input.digit_sample_index)
    def digit_samples():
        df = df_loaded()
        fig = plot_digit(df, input.digit_sample_index())
        return fig

    # ------------ Classification models --------------------------------------

    @render.text
    @reactive.event(input.load, input.dataset, input.standardize, input.split, input.val_split)
    def selected_data():
        _ = compute_splits()
        train_pct = input.split()
        val_pct   = input.val_split()
        test_pct  = max(0, 100 - train_pct - val_pct)
        return (
            f"Dataset: {input.dataset()}\n"
            f"Standardize: {input.standardize()}\n\n"
            f"Training set:   {train_pct:4.1f}%\n"
            f"Validation set: {val_pct:4.1f}%\n"
            f"Test set:       {test_pct:4.1f}%"
        )


    @reactive.effect
    @reactive.event(input.apply_classifier)
    def train_model():
        # ensure fresh splits
        _ = compute_splits()

        # reset model
        model.set(None)

        X_train, Y_train = Xtr(), ytr()
        if X_train is None or Y_train is None:
            return

        # build classifier
        if input.classifier() == "k-NN":
            clf = KNeighborsClassifier(n_neighbors=input.knn_k())
        elif input.classifier() == "Decision Tree":
            clf = DecisionTreeClassifier(
                max_depth=(input.dt_depth() if input.dt_depth() > 0 else None),
                random_state=int(input.classifier_seed())
            )
        elif input.classifier() == "Random Forest":
            clf = RandomForestClassifier(
                n_estimators=input.rf_trees(),
                max_depth=(input.rf_depth() if input.rf_depth() > 0 else None),
                random_state=int(input.classifier_seed())
            )
        else:
            return

        clf.fit(X_train, Y_train)
        with reactive.isolate():
            model.set(clf)
            model_rev.set(data_rev())   # model now corresponds exactly to this data version


    @reactive.effect
    @reactive.event(
        input.dataset, input.split, input.val_split, input.standardize,
        input.use_all_features, input.feature_select
    )
    def _on_data_change():
        _ = compute_splits()

        # increase data version + atomically invalidate model
        with reactive.isolate():
            data_rev.set(data_rev() + 1)
            model.set(None)
            model_rev.set(-1)  # explicitly mark: no model for the current data version


    @reactive.effect
    def update_db_features():
        X_train = Xtr()
        if X_train is None:
            return
        feats = list(X_train.columns)
        ui.update_selectize("db_features", choices=feats)


    
    @render_plotly
    def cm():
        # Version mismatch: ask to retrain
        if model_rev() != data_rev():
            fig = go.Figure()
            fig.add_annotation(
                text="Data changed. Please retrain the model.",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=14),
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            return fig

        m, X_test, Y_test = model(), Xte(), yte()
        if m is None or X_test is None or Y_test is None:
            fig = go.Figure()
            fig.add_annotation(
                text="Train a model to see confusion matrix",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=14),
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            return fig

        # Predictions
        y_pred = m.predict(X_test)

        # Normalization selection
        choice_map = {"Off": None, "Row %": "true", "Column %": "pred", "Overall %": "all"}
        choice = input.cm_normalize()
        normalize = choice_map.get(choice, None)

        # Compute confusion matrix
        labels = list(getattr(m, "classes_", []))
        if normalize is None:
            cmx = confusion_matrix(Y_test, y_pred, labels=labels)
            title = "Confusion Matrix"
            z = cmx.astype(float)
            z_text = cmx.astype(int).astype(str)
        else:
            cmx = confusion_matrix(Y_test, y_pred, labels=labels, normalize=normalize)
            norm_label = {"true": "row", "pred": "column", "all": "overall"}[normalize]
            title = f"Confusion Matrix (normalized: {norm_label})"
            z = cmx.astype(float)
            # Show percentages with 2 decimals as text
            z_text = (np.round(cmx * 100, 2)).astype(str)

        # Convert labels to strings for axes
        tick_labels = [str(c) for c in labels]

        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=tick_labels,      # predicted
                y=tick_labels,      # true
                colorscale="Blues",
                colorbar=dict(title=("Count" if normalize is None else "Proportion")),
                showscale=False,
                text=z_text,
                texttemplate="%{text}",
                hovertemplate=(
                    "True: %{y}<br>"
                    "Pred: %{x}<br>"
                    "Value: %{z:.4f}<extra></extra>"
                    if normalize is not None
                    else "True: %{y}<br>"
                         "Pred: %{x}<br>"
                         "Count: %{z:.0f}<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Predicted label",
            yaxis_title="True label",
            yaxis_autorange="reversed",  # so matrix matches sklearn orientation
            margin=dict(l=80, r=20, t=60, b=60),
        )

        return fig


    @render.text
    def class_report():
        if model_rev() != data_rev():
            return "Data changed. Please retrain the model."
        m, X_test, Y_test = model(), Xte(), yte()
        if m is None or X_test is None or Y_test is None:
            return "Train a model to see classification report"

        y_pred = m.predict(X_test)
        acc = accuracy_score(Y_test, y_pred)
        return f"Accuracy: {acc:.4f}\n\n{classification_report(Y_test, y_pred)}"
    
    
    

    @render_plotly
    def knn_k_sweep():
        # Version mismatch
        if model_rev() != data_rev():
            fig = go.Figure()
            fig.add_annotation(
                text="Data changed. Please retrain the model.",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False,
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            return fig

        if input.classifier() != "k-NN":
            fig = go.Figure()
            fig.add_annotation(
                text="Select k-NN classifier to see the k-sweep.",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False,
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            return fig

        X_train, X_val = Xtr(), Xva()
        y_train, y_val = ytr(), yva()
        if X_train is None or y_train is None:
            fig = go.Figure()
            fig.add_annotation(
                text="Train a model to see the k-NN k-sweep.",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False,
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            return fig

        kmin, kmax = input.knn_k_range()
        k_values = range(int(kmin), int(kmax) + 1)

        return plot_knn_k_range_plotly(
            X_train, y_train,
            X_val, y_val,
            k_values,
            plot_train=True,
        )


    @render_plotly
    def rf_trees_sweep():
        if model_rev() != data_rev():
            fig = go.Figure()
            fig.add_annotation(
                text="Data changed. Please retrain the model.",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False,
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            return fig

        if input.classifier() != "Random Forest":
            fig = go.Figure()
            fig.add_annotation(
                text="Select Random Forest classifier to see the trees sweep.",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False,
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            return fig

        X_train, X_val = Xtr(), Xva()
        y_train, y_val = ytr(), yva()
        if X_train is None or y_train is None:
            fig = go.Figure()
            fig.add_annotation(
                text="Train a model to see the Random Forest trees sweep.",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False,
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            return fig

        tmin, tmax = input.rf_trees_range()
        # fine grained sweeps
        n_values = range(int(tmin), int(tmax) + 1)

        return plot_rf_n_trees_plotly(
            X_train, y_train,
            X_val, y_val,
            n_values,
            plot_train=True,
        )


    @render_plotly
    def rf_depth_sweep():
        if model_rev() != data_rev():
            fig = go.Figure()
            fig.add_annotation(
                text="Data changed. Please retrain the model.",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False,
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            return fig

        if input.classifier() != "Random Forest":
            fig = go.Figure()
            fig.add_annotation(
                text="Select Random Forest classifier to see the depth sweep.",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False,
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            return fig

        X_train, X_val = Xtr(), Xva()
        y_train, y_val = ytr(), yva()
        if X_train is None or y_train is None:
            fig = go.Figure()
            fig.add_annotation(
                text="Train a model to see the Random Forest depth sweep.",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False,
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            return fig

        dmin, dmax = input.rf_depth_range()
        d_values = range(int(dmin), int(dmax) + 1)

        return plot_rf_max_depth_plotly(
            X_train, y_train,
            X_val, y_val,
            d_values,
            plot_train=True,
        )


    @render_plotly
    def dt_depth_sweep():
        if model_rev() != data_rev():
            fig = go.Figure()
            fig.add_annotation(
                text="Data changed. Please retrain the model.",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False,
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            return fig

        if input.classifier() != "Decision Tree":
            fig = go.Figure()
            fig.add_annotation(
                text="Select Decision Tree classifier to see the depth sweep.",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False,
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            return fig

        X_train, X_val = Xtr(), Xva()
        y_train, y_val = ytr(), yva()
        if X_train is None or y_train is None:
            fig = go.Figure()
            fig.add_annotation(
                text="Train a model to see the Decision Tree depth sweep.",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False,
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            return fig

        dmin, dmax = input.dt_depth_range()
        d_values = range(int(dmin), int(dmax) + 1)

        return plot_dt_max_depth_plotly(
            X_train, y_train,
            X_val, y_val,
            d_values,
            plot_train=True,
        )

    
    @render_plotly
    def decision_boundary():
        m = model()
        X_train = Xtr()
        y_train = ytr()

        feats = input.db_features()

        fig = go.Figure()

        # 1) Basic sanity checks
        if model_rev() != data_rev():
            fig.add_annotation(
                text="Data changed. Please retrain the model.",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False,
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            return fig

        if m is None or X_train is None or y_train is None:
            fig.add_annotation(
                text="Train a model first to show the decision boundary.",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False,
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            return fig

        if not feats or len(feats) != 2:
            fig.add_annotation(
                text="Select exactly two features in the sidebar.",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False,
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            return fig

        feat_x, feat_y = feats[0], feats[1]

        return plot_decision_boundaries_plotly(
            classifier=m,
            X=X_train,
            y=y_train,
            feature_pair=[feat_x, feat_y],
            plot_range_percentage=0.05,
        )

    
    # syncs slider with number of features in training data
    @reactive.effect
    def _fi_update_slider_max():
        X_train = Xtr()
        if X_train is None:
            return
        n_feats = len(X_train.columns)
        # Clamp current value to new max
        cur_val = input.fi_topn() if input.fi_topn() <= n_feats else n_feats
        ui.update_slider("fi_topn", max=n_feats if n_feats >= 1 else 1, value=max(1, cur_val))

    
    @render_plotly
    def feat_importance():
        # Version mismatch
        if model_rev() != data_rev():
            fig = go.Figure()
            fig.add_annotation(
                text="Data changed. Please retrain the model.",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=14),
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            return fig

        m, X_train, X_test, Y_test = model(), Xtr(), Xte(), yte()

        # No trained model yet
        if m is None or X_train is None:
            fig = go.Figure()
            fig.add_annotation(
                text=(
                    "Train a model to see feature importances.\n"
                    "• Model-based: Decision Tree / Random Forest\n"
                    "• Permutation: works with any classifier"
                ),
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=13),
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            return fig

        method = input.fi_method()

        # ---------- Model-based feature importance ----------
        if method == "Model-based":
            if not hasattr(m, "feature_importances_"):
                fig = go.Figure()
                fig.add_annotation(
                    text=(
                        "Model-based importances are only available for tree models.\n"
                        "Switch to 'Permutation' or choose Decision Tree / Random Forest."
                    ),
                    x=0.5, y=0.5, xref="paper", yref="paper",
                    showarrow=False, font=dict(size=13),
                )
                fig.update_xaxes(visible=False)
                fig.update_yaxes(visible=False)
                return fig

            s = pd.Series(m.feature_importances_, index=list(X_train.columns))

            # Sort and take top N
            s = s.sort_values(ascending=False)
            topn = max(1, min(input.fi_topn(), len(s)))
            s = s.head(topn)

            # Normalize if requested
            if input.fi_normalize() and s.sum() != 0:
                s = s / s.sum()

            # Safety: remove non-finite
            s = s.replace([np.inf, -np.inf], np.nan).dropna()
            if s.empty:
                fig = go.Figure()
                fig.add_annotation(
                    text="No finite feature importances to display.",
                    x=0.5, y=0.5, xref="paper", yref="paper",
                    showarrow=False, font=dict(size=13),
                )
                fig.update_xaxes(visible=False)
                fig.update_yaxes(visible=False)
                return fig

            # Horizontal bar: most important on top
            feat_names = list(s.index[::-1])
            values = s.values[::-1]

            fig = go.Figure(
                data=go.Bar(
                    x=values,
                    y=feat_names,
                    orientation="h",
                    hovertemplate="Feature=%{y}<br>Importance=%{x:.4f}<extra></extra>",
                )
            )

            fig.update_layout(
                title="Model-based Feature Importances"
                      + (f" (top {topn})" if topn < len(X_train.columns) else ""),
                xaxis_title="Importance" + (" (normalized)" if input.fi_normalize() else ""),
                yaxis_title="Feature",
                margin=dict(l=120, r=20, t=60, b=60),
            )
            return fig

        # ---------- Permutation feature importance ----------
        if X_test is None or Y_test is None:
            fig = go.Figure()
            fig.add_annotation(
                text="No test split available for permutation importance.",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=13),
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            return fig

        n_rep = max(1, int(input.fi_repeats()))
        scoring = input.fi_scoring()

        try:
            result = permutation_importance(
                m, X_test, Y_test,
                n_repeats=n_rep,
                random_state=int(input.classifier_seed()),
                scoring=scoring,
            )
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Permutation importance failed:\n{e}",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=12),
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            return fig

        means = pd.Series(np.abs(result.importances_mean), index=list(X_test.columns))

        # Sort, top N
        means = means.sort_values(ascending=False)
        topn = max(1, min(input.fi_topn(), len(means)))
        s = means.head(topn)

        # Normalize if requested
        if input.fi_normalize() and s.sum() != 0:
            s = s / s.sum()

        # Remove non-finite
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No finite permutation importances to display.",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=13),
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            return fig

        feat_names = list(s.index[::-1])
        values = s.values[::-1]

        fig = go.Figure(
            data=go.Bar(
                x=values,
                y=feat_names,
                orientation="h",
                hovertemplate=(
                    "Feature=%{y}<br>"
                    f"Permutation importance (abs Δ{scoring})=%{{x:.4f}}<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            title=f"Permutation Importance (top {topn}, {n_rep} repeats)",
            xaxis_title=f"Permutation importance (abs Δ{scoring})"
                        + (" (normalized)" if input.fi_normalize() else ""),
            yaxis_title="Feature",
            margin=dict(l=140, r=20, t=60, b=60),
        )

        return fig



    @reactive.effect
    def _curve_update_class_choices():
        m, Y_test = model(), yte()
        if m is None or Y_test is None:
            return
        classes_raw = list(getattr(m, "classes_", [])) or list(pd.Series(Y_test).unique())
        class_labels = [str(c) for c in classes_raw]  # <-- stringify to avoid numpy types
        ui.update_select(
            "curve_class",
            choices=class_labels,
            selected=(class_labels[0] if class_labels else None),
        )

    

    @render_plotly
    def curve_plot():
        # check version match
        if model_rev() != data_rev():
            fig = go.Figure()
            fig.add_annotation(
                text="Data changed. Please retrain the model.",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=14)
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            return fig

        m, X_test, Y_test = model(), Xte(), yte()
        fig = go.Figure()

        if m is None or X_test is None or Y_test is None:
            fig.add_annotation(
                text="Train a model to view curves.",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=14)
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            return fig

        metric = input.curve_metric()
        classes = list(getattr(m, "classes_", []))
        labels  = [str(c) for c in classes]  # for UI

        # Get scores (probabilities or decision_function)
        if hasattr(m, "predict_proba"):
            scores = m.predict_proba(X_test)
        elif hasattr(m, "decision_function"):
            s = m.decision_function(X_test)
            if s.ndim == 1:  # binär, 1D Score
                s_norm = (s - s.min()) / (s.max() - s.min() + 1e-12)
                scores = np.vstack([1 - s_norm, s_norm]).T
            else:
                scores = s
        else:
            fig.add_annotation(
                text="Classifier has neither predict_proba nor decision_function.",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=14)
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            return fig

        y = np.array(Y_test)

        # ---------- binary classification ----------
        if len(classes) == 2:
            pos_idx = 1
            pos_cls = classes[pos_idx]
            s_pos   = scores[:, pos_idx]
            y_bin   = (y == pos_cls).astype(int)

            if metric == "ROC":
                fpr, tpr, thr = roc_curve(y_bin, s_pos)
                auc_val = auc(fpr, tpr)

                # Keep only finite values (no inf/NaN for JSON!)
                mask = np.isfinite(fpr) & np.isfinite(tpr) & np.isfinite(thr)
                fpr, tpr, thr = fpr[mask], tpr[mask], thr[mask]

                if fpr.size == 0:
                    fig = go.Figure()
                    fig.add_annotation(
                        text="Could not compute a finite ROC curve (check labels/scores).",
                        x=0.5, y=0.5, xref="paper", yref="paper",
                        showarrow=False, font=dict(size=14),
                    )
                    fig.update_xaxes(visible=False)
                    fig.update_yaxes(visible=False)
                    return fig


                # customdata: [threshold, TPR, FPR]
                customdata = np.stack([thr, tpr, fpr], axis=-1)

                fig.add_trace(
                    go.Scatter(
                        x=fpr,
                        y=tpr,
                        mode="lines+markers",
                        name=f"Classifier (AUC={auc_val:.3f})",
                        customdata=customdata,
                        hovertemplate=(
                            "Threshold=%{customdata[0]:.3f}<br>"
                            "TPR=%{customdata[1]:.3f}<br>"
                            "FPR=%{customdata[2]:.3f}<extra></extra>"
                        ),
                    )
                )

                # Chance line
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode="lines",
                        name="Chance level (AUC=0.5)",
                        line=dict(dash="dash"),
                        hoverinfo="skip",
                    )
                )

                fig.update_layout(
                    title=f"ROC Curve (positive class: {pos_cls})",
                    xaxis_title="False Positive Rate (FPR)",
                    yaxis_title="True Positive Rate (TPR)",
                )

            else:  # PR
                prec, rec, thr = precision_recall_curve(y_bin, s_pos)
                ap = average_precision_score(y_bin, s_pos)

                if thr.size > 0:
                    thr_full = np.r_[thr, thr[-1]]
                else:
                    # some finite default, e.g. 0.5
                    thr_full = np.full_like(prec, 0.5, dtype=float)

                # Keep only finite values
                mask = np.isfinite(prec) & np.isfinite(rec) & np.isfinite(thr_full)
                prec, rec, thr_full = prec[mask], rec[mask], thr_full[mask]

                if prec.size == 0:
                    fig = go.Figure()
                    fig.add_annotation(
                        text="Could not compute a finite PR curve (check labels/scores).",
                        x=0.5, y=0.5, xref="paper", yref="paper",
                        showarrow=False, font=dict(size=14),
                    )
                    fig.update_xaxes(visible=False)
                    fig.update_yaxes(visible=False)
                    return fig

                customdata = np.stack([thr_full, prec, rec], axis=-1)


                fig.add_trace(
                    go.Scatter(
                        x=rec,
                        y=prec,
                        mode="lines+markers",
                        name=f"Classifier (AP={ap:.3f})",
                        customdata=customdata,
                        hovertemplate=(
                            "Threshold=%{customdata[0]:.3f}<br>"
                            "Precision=%{customdata[1]:.3f}<br>"
                            "Recall=%{customdata[2]:.3f}<extra></extra>"
                        ),
                    )
                )

                pos_rate = (y_bin == 1).mean()
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[pos_rate, pos_rate],
                        mode="lines",
                        name=f"Chance level (AP≈{pos_rate:.3f})",
                        line=dict(dash="dash"),
                        hoverinfo="skip",
                    )
                )

                fig.update_layout(
                    title=f"Precision–Recall Curve (positive class: {pos_cls})",
                    xaxis_title="Recall",
                    yaxis_title="Precision",
                )

        # ---------- Multiclass: One-vs-Rest ----------
        else:
            sel_label = input.curve_class()
            if not sel_label or sel_label not in labels:
                fig.add_annotation(
                    text="Select a class.",
                    x=0.5, y=0.5, xref="paper", yref="paper",
                    showarrow=False, font=dict(size=14)
                )
                fig.update_xaxes(visible=False)
                fig.update_yaxes(visible=False)
                return fig

            idx = labels.index(sel_label)
            cls = classes[idx]
            y_bin = (y == cls).astype(int)
            s_cls = scores[:, idx]

            if metric == "ROC":
                fpr, tpr, thr = roc_curve(y_bin, s_cls)
                auc_val = auc(fpr, tpr)

                mask = np.isfinite(fpr) & np.isfinite(tpr) & np.isfinite(thr)
                fpr, tpr, thr = fpr[mask], tpr[mask], thr[mask]

                if fpr.size == 0:
                    fig = go.Figure()
                    fig.add_annotation(
                        text="Could not compute a finite ROC curve for this class.",
                        x=0.5, y=0.5, xref="paper", yref="paper",
                        showarrow=False, font=dict(size=14),
                    )
                    fig.update_xaxes(visible=False)
                    fig.update_yaxes(visible=False)
                    return fig


                customdata = np.stack([thr, tpr, fpr], axis=-1)

                fig.add_trace(
                    go.Scatter(
                        x=fpr,
                        y=tpr,
                        mode="lines+markers",
                        name=f"Classifier (AUC={auc_val:.3f})",
                        customdata=customdata,
                        hovertemplate=(
                            "Threshold=%{customdata[0]:.3f}<br>"
                            "TPR=%{customdata[1]:.3f}<br>"
                            "FPR=%{customdata[2]:.3f}<extra></extra>"
                        ),
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode="lines",
                        name="Chance level (AUC=0.5)",
                        line=dict(dash="dash"),
                        hoverinfo="skip",
                    )
                )

                fig.update_layout(
                    title=f"ROC Curve (one-vs-rest, class {sel_label})",
                    xaxis_title="False Positive Rate (FPR)",
                    yaxis_title="True Positive Rate (TPR)",
                )

            else:  # PR
                prec, rec, thr = precision_recall_curve(y_bin, s_cls)
                ap = average_precision_score(y_bin, s_cls)

                if thr.size > 0:
                    thr_full = np.r_[thr, thr[-1]]
                else:
                    thr_full = np.full_like(prec, 0.5, dtype=float)

                mask = np.isfinite(prec) & np.isfinite(rec) & np.isfinite(thr_full)
                prec, rec, thr_full = prec[mask], rec[mask], thr_full[mask]

                if prec.size == 0:
                    fig = go.Figure()
                    fig.add_annotation(
                        text="Could not compute a finite PR curve for this class.",
                        x=0.5, y=0.5, xref="paper", yref="paper",
                        showarrow=False, font=dict(size=14),
                    )
                    fig.update_xaxes(visible=False)
                    fig.update_yaxes(visible=False)
                    return fig

                customdata = np.stack([thr_full, prec, rec], axis=-1)

                fig.add_trace(
                    go.Scatter(
                        x=rec,
                        y=prec,
                        mode="lines+markers",
                        name=f"Classifier (AP={ap:.3f})",
                        customdata=customdata,
                        hovertemplate=(
                            "Threshold=%{customdata[0]:.3f}<br>"
                            "Precision=%{customdata[1]:.3f}<br>"
                            "Recall=%{customdata[2]:.3f}<extra></extra>"
                        ),
                    )
                )

                pos_rate = (y_bin == 1).mean()
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[pos_rate, pos_rate],
                        mode="lines",
                        name=f"Chance level (AP≈{pos_rate:.3f})",
                        line=dict(dash="dash"),
                        hoverinfo="skip",
                    )
                )

                fig.update_layout(
                    title=f"Precision–Recall Curve (one-vs-rest, class {sel_label})",
                    xaxis_title="Recall",
                    yaxis_title="Precision",
                )

        # Common layout settings
        fig.update_xaxes(range=[-0.01, 1.01])
        fig.update_yaxes(
            range=[-0.01, 1.01],
        )
        fig.update_layout(
            autosize=True,
            margin=dict(l=60, r=20, t=40, b=60),
            legend=dict(x=0.01, y=0.99),
        )

        return fig



    @render.data_frame
    def errors_table():
        m, X_test, Y_test = model(), Xte(), yte()
        if m is None or X_test is None or Y_test is None:
            return pd.DataFrame({"Info": ["Train a model to see misclassifications"]})

        y_true = pd.Series(Y_test, name="y_true").reset_index(drop=True)
        y_pred = pd.Series(m.predict(X_test), name="y_pred").reset_index(drop=True)
        # dfX = pd.DataFrame(X_test).reset_index(drop=True)
        out = pd.concat([y_true, y_pred], axis=1) # removed X_test to reduce width
        wrong = out[y_true != y_pred].copy()
        if wrong.empty:
            return pd.DataFrame({"Info": ["No misclassifications"]})

        # add predicted probability for predicted class if available
        if hasattr(m, "predict_proba"):
            proba = pd.DataFrame(m.predict_proba(X_test), columns=[f"p({c})" for c in getattr(m, "classes_", [])])
            proba = proba.reset_index(drop=True)
            wrong = pd.concat([wrong.reset_index(drop=True), proba.loc[wrong.index].reset_index(drop=True)], axis=1)

        # sort by confidence descending (most confident wrong first) if proba exists
        prob_cols = [c for c in wrong.columns if c.startswith("p(")]
        if prob_cols:
            cls_to_col = {c: f"p({c})" for c in getattr(m, "classes_", [])}
            wrong["p_pred"] = wrong.apply(lambda r: r[cls_to_col.get(r["y_pred"], prob_cols[0])], axis=1)
            wrong = wrong.sort_values("p_pred", ascending=False)
        return wrong.round(3)
    
    @render.text
    @reactive.event(input.load, input.dataset, input.split, input.val_split,
                    input.standardize, input.use_all_features, input.feature_select)
    def data_summary():
        _ = compute_splits()
        df = df_loaded()
        n_rows, n_cols = df.shape
        n_missing = df.isnull().sum().sum()
        n_num = df.select_dtypes(include=[np.number]).shape[1]
        n_cat = df.select_dtypes(include=['object', 'category']).shape[1]

        target_col = DATASETS[input.dataset()][1]
        target_counts = df[target_col].value_counts()

        def _none_to_str(x): return "—" if x is None else str(x.shape[0])
        train_size = _none_to_str(Xtr())
        val_size   = _none_to_str(Xva())
        test_size  = _none_to_str(Xte())


        return (f"Rows: {n_rows}\n"
                f"Columns: {n_cols} (Numeric: {n_num}, Categorical: {n_cat})\n\n"
                f"Missing values: {n_missing}\n\n"
                f"Target class distribution:\n{target_counts.to_string()}\n\n"
                f"Train set size: {train_size}\n"
                f"Validation set size: {val_size}\n"
                f"Test set size: {test_size}"
                )
    
    @render.text
    def split_overview():
        train, val, test = split_percentages()
        return f"Train: {train:4.1f}%\nVal:   {val:4.1f}%\nTest:  {test:4.1f}%"
    
    @render.text
    def pca_explained():
        if input.load() == 0:
            return "Click 'Load dataset' to see PCA explained variance"
        df = df_loaded()
        tgt = DATASETS[input.dataset()][1]
        feats = [c for c in df.select_dtypes(include=[np.number]).columns if c != tgt]
        if len(feats) < 2:
            return "Need at least two numeric features for PCA"
        X = df[feats].copy()
        if input.standardize():
            X[:] = StandardScaler().fit_transform(X)
        pca = PCA().fit(X)
        var_ratios = pca.explained_variance_ratio_
        cum_var = np.cumsum(var_ratios)
        lines = ["PC\tExplained Var.\tCumulative Var."]
        for i, (var, cum) in enumerate(zip(var_ratios, cum_var), start=1):
            lines.append(f"{i}\t{var:.4f}\t\t{cum:.4f}")
            if i == 2: break
        return "\n".join(lines)
    
    @render.text
    def pca_loadings_top_features():
        if input.load() == 0:
            return "Click 'Load dataset' to see PCA loadings"
        df = df_loaded()
        tgt = DATASETS[input.dataset()][1]
        feats = [c for c in df.select_dtypes(include=[np.number]).columns if c != tgt]
        if len(feats) < 2:
            return "Need at least two numeric features for PCA"
        X = df[feats].copy()
        if input.standardize():
            X[:] = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2).fit(X)
        loadings = pd.DataFrame(pca.components_.T, index=feats, columns=["PC1", "PC2"])
        topn = min(5, len(feats))
        lines = ["Top feature loadings per PC:"]
        for pc in ["PC1", "PC2"]:
            lines.append(f"\n{pc}:")
            sorted_loadings = loadings[pc].abs().sort_values(ascending=False).head(topn)
            for feat in sorted_loadings.index:
                val = loadings.loc[feat, pc]
                lines.append(f"  {feat:<25}: {val:7.4f}")
        return "\n".join(lines)

app = App(app_ui, server, static_assets=Path(__file__).parent / "www")

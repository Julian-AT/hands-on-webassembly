from shiny import App, ui, render, reactive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from io import StringIO, BytesIO
import json

import torch
import torch.nn as nn
import u5_utils as U5

from shinywidgets import output_widget, render_plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sns.set_theme()


app_ui = ui.page_fluid(

    ui.tags.head(
        ui.tags.link(
            rel="stylesheet",
            href="https://cdn.jsdelivr.net/npm/bootswatch@5.3.0/dist/zephyr/bootstrap.min.css"
        ),
        ui.tags.style("""
        /* Remove border around nav tabs and sidebar panels */
        .nav-tabs {
            border: none !important;
            box-shadow: none !important;
        }
        /* Optional: remove background highlight so it blends fully */
        .tab-content {
            background-color: transparent !important;
        }
        """),
        ui.tags.style("""
            /* Remove border around nav tabs and sidebar panels */
            .nav-tabs {
                border: none !important;
                box-shadow: none !important;
            }
            .tab-content {
                background-color: transparent !important;
            }

            /* Generic container that fills its parent (card) */
            .plot-container {
                width: 100%;
                height: 100%;
                display: flex;
                justify-content: center;
                align-items: center;
                margin-left: auto;
                margin-right: auto;
            }

            /* First child = output_widget's outer <div> */
            .plot-container > * {
                width: 100%;
                height: 100%;
            }

            /* Plotly's inner div */
            .plot-container > * > div {
                width: 100%;
                height: 100%;
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
    ),

    ui.h2("Hands-on AI I - Unit 5: Logistic Regression", class_="text-center mb-4"),
    ui.navset_tab(
        ui.nav_panel(
            "Linear Regression",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h4("Data"),
                    ui.input_radio_buttons(id="pr_choice", label= "",choices=["Custom", "Random", "Classes"], selected="Custom"),
                    ui.input_numeric("pr_seed", "Seed", 42, min=0),
                    ui.input_numeric("pr_n", "Number of samples", 200, min=5, max=50_000),
                    ui.panel_conditional("input.pr_choice!='Classes'",
                        ui.input_numeric("pr_var", "Variance", 0.5, min=0.0, step=0.1),
                    ),
                    ui.panel_conditional("input.pr_choice=='Custom'",
                        ui.input_text("pr_true_c", "Coefficients", "0.422;0.241")
                    ),
                    ui.input_action_button("pr_make", "Generate data", class_="btn-primary"),

                    ui.hr(),
                    ui.h4("Model"),
                    ui.input_slider("pr_deg", "Polynomial degree", min=0, max=10, value=1),
                    ui.input_checkbox("pr_fit", "Show fitted model"),
                    ui.panel_conditional("input.pr_choice=='Custom'",
                        ui.input_checkbox("pr_true_coefs", "Show true model"),
                    ),
                    ui.hr(),
                    ui.download_button("pr_dl_coeffs", "Download coefficients (.json)", class_="btn-secondary"),
                    ui.download_button("pr_dl_data", "Download dataset (.csv)", class_="btn-secondary"),
                ),
                ui.navset_card_pill(
                    ui.nav_panel("Plot",
                        ui.card(
                            ui.div(
                                output_widget("pr_plot_model", width="100%", height="100%"),
                                class_="plot-container"
                            ),
                        height="400px")
                    ),
                    ui.nav_panel("Data (head)",
                        ui.output_data_frame("pr_head")
                    )
                ),
                ui.card(
                    ui.card_header("Results"),
                    ui.output_text_verbatim("pr_fitted_coeffs"),
                    ui.output_text_verbatim("pr_mse")
                )            
            ),
        ),
        ui.nav_panel(
            "Logistic Regression (1D)",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h4("Data"),
                    ui.input_numeric("lc_seed", "Seed", 42),
                    ui.input_numeric("lc_n", "Samples", 200, min=10, max=50_000),
                    ui.input_action_button("lc_make", "Generate data", class_="btn-primary"),
                    ui.hr(),
                    ui.h4("Training (softmax / CE)"),
                    ui.input_numeric("lc_epochs", "Epochs", 20, min=1),
                    ui.input_numeric("lc_lr", "Learning rate", 0.1, min=1e-5, step=0.05),
                    ui.input_numeric("lc_mom", "Momentum", 0.0, min=0.0, max=0.99, step=0.05),
                    ui.input_action_button("lc_train", "Train linear classifier", class_="btn-primary"),
                    ui.hr(),
                    ui.download_button("lc_dl_coeffs", "Download coefficients (.json)", class_="btn-secondary"),
                    ui.download_button("lc_dl_data", "Download dataset (.csv)", class_="btn-secondary"),
                ),
                ui.navset_card_pill(
                    ui.nav_panel("Data & Decision Function",
                        ui.card(
                            ui.div(
                                output_widget("lc_plot", width="100%", height="100%"),
                                class_="plot-container",
                            ),
                            height="420px",
                        ),
                    ),
                    ui.nav_panel("Illustration: Sigmoid function",
                        ui.card(
                            ui.div(
                                output_widget("lc_sigmoid", width="100%", height="100%"),
                                class_="plot-container",
                            ),
                            height="420px",
                        ),
                    ),
                ),
                ui.card(
                    ui.card_header("Results"),
                    ui.output_text_verbatim("lc_coeffs_txt"),
                    ui.output_text("lc_metrics"),
                ),
            ),
        ),
        ui.nav_panel("Logistic Regression (2D)",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h4("Data"),
                    ui.input_numeric("lc_seed_2d", "Seed", 42),
                    ui.input_file("lc_file_2d", "Upload CSV", multiple=False, accept=[".csv"]),
                    ui.input_checkbox("lc_ignore_header_2d", "Ignore header (auto-name columns)"),
                    ui.input_action_button("lc_make_2d", "Load data", class_="btn-primary"),
                    ui.hr(),
                    ui.h4("Train/Test split"),
                    ui.input_numeric("lc_split_2d", "Train fraction", 0.8, min=0.1, max=0.9, step=0.05),
                    ui.hr(),
                    ui.h4("Training"),
                    ui.input_numeric("lc_epochs_2d", "Iterations", 20, min=1),
                    ui.input_numeric("lc_lr_2d", "Learning rate", 0.1, min=1e-5, step=0.05),
                    ui.input_numeric("lc_mom_2d", "Momentum", 0.0, min=0.0, max=0.99, step=0.05),
                    ui.input_action_button("lc_train_2d", "Train classifier", class_="btn-primary"),
                ),
                ui.card(
                    ui.card_header("Ground truth"),
                    ui.div(
                        output_widget("lc_plot_2d", width="100%", height="100%"),
                        class_="plot-container"
                    ),
                height="500px"),
                ui.card(
                    ui.card_header("Results"),
                    ui.div(
                        output_widget("lc_plot_prect_2d", width="100%", height="100%"),
                        class_="plot-container"
                    ),
                    ui.output_text_verbatim("lc_coeffs_txt_2d"),
                    ui.output_text_verbatim("lc_metrics_2d"),
                height="420px"),
            ),
        ),
        ui.nav_panel(
            "MNIST / Fashion-MNIST",
            ui.tags.div(style="margin-top: 20px;"),
            ui.card(
                ui.card_header("About the MNIST and Fashion-MNIST datasets"),
                ui.tags.p(
                    "The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28x28 pixels. "
                    "the Fashion-MNIST dataset is a similar dataset containing images of clothing items, also in grayscale and of the same size. ",
                    "Both datasets are already split into a training set of 60,000 images and a test set of 10,000 images. "
                ),
            ),
            ui.card(
                ui.card_header("MNIST"),
                ui.div(
                    ui.output_plot("mnist_samples", width="100%", height="100%"),
                    class_="plot-container"
                ),
                height="420px"
            ),
            # bit of vertical space
            ui.tags.br(),
            ui.card(
                ui.card_header("Fashion-MNIST"),
                ui.div(
                    ui.output_plot("fashionmnist_samples", width="100%", height="100%"),
                    class_="plot-container"
                ),
                height="420px"
            ),
        ),
        ui.nav_panel(
            "Linear Classifier on MNIST",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h4("Dataset"),
                    ui.input_select(
                        "mn_ds",
                        "Dataset",
                        {"mnist": "MNIST", "fashion": "Fashion-MNIST"},
                        selected="mnist",
                    ),
                    ui.input_numeric("mn_seed", "Seed", 42, min=0),
                    ui.input_numeric("mn_batch", "Batch size", 256, min=8, max=1024),
                    ui.hr(),
                    ui.h4("Training"),
                    ui.input_numeric("mn_epochs", "Epochs", 3, min=1),
                    ui.input_numeric("mn_lr", "Learning rate", 0.1, min=1e-5, step=0.05),
                    ui.input_numeric("mn_mom", "Momentum", 0.9, min=0.0, max=0.99, step=0.05),
                    ui.input_action_button("mn_train", "Train", class_="btn-primary"),
                    ui.output_ui("mn_progress"),
                    ui.hr(),
                    ui.h4("Evaluation"),
                    ui.input_action_button("mn_eval", "Evaluate", class_="btn-primary"),
                    ui.input_checkbox("mn_invert", "Invert pixels"),
                    ui.input_checkbox("mn_hflip", "Horizontal flip"),
                    
                ),
                ui.card(
                    ui.card_header("Sample predictions on test set"),
                    ui.output_plot("mn_examples", height="420px"),
                ),
                ui.card(
                    ui.card_header("Evaluation on test set"),
                    ui.output_text("mn_acc"),
                    ui.div(
                        output_widget("mn_cm", width="100%", height="100%"),
                        class_="plot-container",
                    ),
                    height="500px",
                ),
            ),
        ),
        ui.nav_panel(
            "Misclassification analysis",
            ui.tags.div(style="margin-top: 20px;"),
            ui.card(
                ui.card_header("Misclassified examples with softmax probabilities"),
                ui.output_ui("mn_mis_table"),
            ),
        ),
    ),
)

# -----------------------------------------------------------------------------
# Server helpers
# -----------------------------------------------------------------------------
def _df_head(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    try:
        return df.head(n)
    except Exception:
        return pd.DataFrame()


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))

def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


# -----------------------------------------------------------------------------
# Server
# -----------------------------------------------------------------------------
def server(input, output, session):

    # -------------------- Polynomial Regression ------------------------------
    pr_df = reactive.Value(pd.DataFrame())
    pr_coeffs = reactive.Value(None)
    pr_true_coefficients = reactive.Value(None)


    @reactive.effect
    @reactive.event(input.pr_make)
    def _pr_make():
        U5.set_seed(int(input.pr_seed()))
        choice = input.pr_choice()

        if choice == "Random":
            df = U5.get_dataset_unknown(int(input.pr_n()), float(input.pr_var()))
            pr_true_coefficients.set(None)  # no true model here

        elif choice == "Custom":
            true_coeffs = [float(x) for x in input.pr_true_c().split(";")]
            pr_true_coefficients.set(true_coeffs)
            df = U5.get_dataset(int(input.pr_n()), float(input.pr_var()), true_coeffs)

        elif choice == "Classes":
            df = U5.get_dataset_logistic(num_pairs=input.pr_n())
            pr_true_coefficients.set(None)  # again, no polynomial true model

        pr_df.set(df)

        

    @reactive.effect
    @reactive.event(input.pr_fit, input.pr_make, input.pr_deg)
    def _pr_fit():
        df = pr_df.get()
        if df.empty:
            return
        deg = int(input.pr_deg())
        coeffs = U5.minimize_mse(df, degree=deg)
        pr_coeffs.set(coeffs)

    @output
    @render.text
    def pr_fitted_coeffs():
        coeffs = pr_coeffs.get()
        if coeffs is None:
            return "(No coefficients yet — click 'Generate data' or 'Fit model')"
        return json.dumps({"coefficients": coeffs}, indent=2)
 

    @output
    @render_plotly
    @reactive.event(input.pr_make, input.pr_fit, input.pr_true_coefs, input.pr_deg, input.pr_choice)
    def pr_plot_model():
        df = pr_df.get()
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="Click 'Generate data' to start",
                x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            fig.update_layout(title="Polynomial regression", autosize=True)
            return fig

        x = df[df.columns[0]].to_numpy()
        y = df[df.columns[1]].to_numpy()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name="Data"
            )
        )

        coeffs = pr_coeffs.get()
        if input.pr_fit() and coeffs is not None:
            x_line = np.linspace(x.min(), x.max(), 300)
            y_line = np.zeros_like(x_line)
            for degree, c in enumerate(coeffs):
                y_line += c * (x_line ** degree)

            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    name="Fitted model"
                )
            )

        true_coeffs = pr_true_coefficients.get()
        if input.pr_choice() == "Custom" and input.pr_true_coefs() and true_coeffs is not None:
            x_line = np.linspace(x.min(), x.max(), 300)
            y_true = np.zeros_like(x_line)
            for degree, c in enumerate(true_coeffs):
                y_true += c * (x_line ** degree)

            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_true,
                    mode="lines",
                    name="True model",
                    line=dict(dash="dash")
                )
            )

        fig.update_layout(
            xaxis_title=df.columns[0],
            yaxis_title=df.columns[1],
            template="plotly_white",
            autosize=True,
        )
        return fig



    @output
    @render.text
    def pr_mse():
        df = pr_df.get()
        coeffs = pr_coeffs.get()
        if df.empty or coeffs is None:
            return ""
        # Evaluate MSE on the training data
        x = df[df.columns[0]].to_numpy()
        y = df[df.columns[1]].to_numpy()
        y_hat = np.zeros_like(x)
        for degree, c in enumerate(coeffs):
            y_hat += c * (x ** degree)
        return f"Training MSE: {_mse(y, y_hat):.4f}"
    
    @output
    @render.text
    def pr_coeffs_txt():
        coeffs = pr_coeffs.get()
        if coeffs is None:
            return "(No coefficients yet — click 'Fit model')"
        return json.dumps({"coefficients": coeffs}, indent=2)

    @output
    @render.data_frame
    def pr_head():
        return _df_head(pr_df.get())

    @render.download(filename=lambda: "u5_polynomial_coeffs.json")
    def pr_dl_coeffs():
        coeffs = pr_coeffs.get() or []
        yield json.dumps({"coefficients": coeffs}, indent=2)

    @render.download(filename=lambda: "u5_polynomial_regression_dataset.csv")
    def pr_dl_data():
        df = pr_df.get()
        out = df.to_csv(index=False)
        yield out

    # -------------------- Logistic Regression --------------------------------
    lc_df = reactive.Value(pd.DataFrame())
    lc_coeffs = reactive.Value(None)
    lc_acc = reactive.Value(None)

    
    @output
    @render_plotly
    def lc_sigmoid():
        x = np.linspace(-10, 10, 400)
        y = _sigmoid(x)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name="σ(x)"
            )
        )

        # Decision threshold lines
        fig.add_shape(
            type="line",
            x0=x.min(), x1=x.max(), y0=0.5, y1=0.5,
            line=dict(dash="dash")
        )
        fig.add_shape(
            type="line",
            x0=0, x1=0, y0=y.min(), y1=y.max(),
            line=dict(dash="dash")
        )

        fig.update_layout(
            title="Sigmoid function",
            xaxis_title="x",
            yaxis_title="σ(x)",
            template="plotly_white"
        )
        return fig


    @reactive.effect
    @reactive.event(input.lc_make)
    def _lc_make():
        U5.set_seed(int(input.lc_seed()))
        df = U5.get_dataset_logistic(int(input.lc_n()))
        lc_df.set(df)
        lc_coeffs.set(None)
        lc_acc.set(None)

    @reactive.effect
    @reactive.event(input.lc_train)
    def _lc_train():
        U5.set_seed(int(input.lc_seed()))

        df = lc_df.get()
        if df.empty:
            return
        coeffs = U5.minimize_ce(
            df,
            iterations=int(input.lc_epochs()),
            learning_rate=float(input.lc_lr()),
            momentum=float(input.lc_mom()),
            use_cuda_if_available=False,
        )
        lc_coeffs.set(coeffs)
        # Evaluate on training set (toy)
        X = df[df.columns[:-1]]
        y = df[df.columns[-1]].to_numpy()
        y_hat = U5.predict_logistic(X, coeffs)
        lc_acc.set(float(np.mean(y_hat == y)))

    


    @output
    @render_plotly
    def lc_plot():
        df = lc_df.get()
        fig = go.Figure()

        if df.empty:
            fig.add_annotation(
                text="Click 'Generate data'",
                x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            return fig

        x_col = df.columns[0]
        y_col = df.columns[1]

        coeffs = lc_coeffs.get()

        # Colors for the points: red and blue
        colors = ['red' if val == 0 else 'blue' for val in df[y_col]]

        # Always show the data points (y ∈ {0,1})
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode="markers",
                name="Data (classes)",
                marker=dict(
                    color=colors,
                    showscale=False
                )
            )
        )

        if coeffs is not None:
            # Build softmax logits for a 1D feature and plot P(class=k | x) for all classes
            x_grid = np.linspace(df[x_col].min(), df[x_col].max(), 300)
            coeffs_list = list(coeffs)

            # coeffs format: tuple( (bias, w1, w2, ...) for each class )
            b = np.array([c[0] for c in coeffs_list])   # (K,)
            W = np.array([c[1:] for c in coeffs_list])  # (K, d)

            X = x_grid.reshape(-1, 1)                   # (n, 1)
            scores = X @ W.T + b                        # (n, K)
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)  # (n, K)

            # Add one curve per class: P(class = k | x)
            num_classes = probs.shape[1]
            for k in range(num_classes):
                fig.add_trace(
                    go.Scatter(
                        x=x_grid,
                        y=probs[:, k],
                        mode="lines",
                        name=f"P(class={k} | x)"
                    )
                )

        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col,
            template="plotly_white",
            yaxis=dict(range=[-0.1, 1.1])
        )
        return fig



    @output
    @render.text
    def lc_coeffs_txt():
        coeffs = lc_coeffs.get()
        if coeffs is None:
            return "(No coefficients yet — click 'Train linear classifier')"
        return json.dumps({"coefficients": coeffs}, indent=2)

    @output
    @render.text
    def lc_metrics():
        acc = lc_acc.get()
        if acc is None:
            return ""
        return f"Training accuracy: {acc*100:.1f}%"

    @render.download(filename=lambda: "u5_logistic_coeffs.json")
    def lc_dl_coeffs():
        coeffs = lc_coeffs.get() or []
        yield json.dumps({"coefficients": coeffs}, indent=2)

    @render.download(filename=lambda: "u5_logistic_dataset.csv")
    def lc_dl_data():
        df = lc_df.get()
        out = df.to_csv(index=False)
        yield out


    # -------------------- Logistic Regression 2D -----------------------------
    lc_df_2d = reactive.Value(pd.DataFrame())
    lc_coeffs_2d = reactive.Value(None)
    lc_acc_2d_test = reactive.Value(None)
    lc_acc_2d_train = reactive.Value(None)
    lc_df_2d_train = reactive.Value(pd.DataFrame())
    lc_df_2d_test = reactive.Value(pd.DataFrame())

    @reactive.effect
    @reactive.event(input.lc_make_2d)
    def _lc_make_2d():
        file = input.lc_file_2d()
        if not file:
            return
        # Read file content
        path = file[0]["datapath"]
        df = U5.get_dataset_from_csv(
            str(path),
            delimiter=",",
            ignore_header=bool(input.lc_ignore_header_2d()),
        )
        lc_df_2d.set(df)
        lc_coeffs_2d.set(None)
        lc_acc_2d_train.set(None)
        lc_acc_2d_test.set(None)

    @reactive.effect
    @reactive.event(input.lc_train_2d)
    def _lc_train_2d():
        df = lc_df_2d.get()
        if df.empty:
            return
        # Split into train/test
        frac = float(input.lc_split_2d())
        n_train = int(len(df) * frac)
        U5.set_seed(int(input.lc_seed_2d()))
        df_train = df.sample(n=n_train)#, random_state=int(input.lc_seed_2d()))
        df_test = df.drop(df_train.index)
        lc_df_2d_train.set(df_train)
        lc_df_2d_test.set(df_test)
        coeffs = U5.minimize_ce(
            df_train,
            iterations=int(input.lc_epochs_2d()),
            learning_rate=float(input.lc_lr_2d()),
            momentum=float(input.lc_mom_2d()),
            use_cuda_if_available=False,
        )
        lc_coeffs_2d.set(coeffs)
        # Evaluate on test set
        X_test = df_test[df_test.columns[:-1]]
        y_test = df_test[df_test.columns[-1]].to_numpy()
        y_hat = U5.predict_logistic(X_test, coeffs)
        lc_acc_2d_test.set(float(np.mean(y_hat == y_test)))
        # Evaluate on train set
        X_train = df_train[df_train.columns[:-1]]
        y_train = df_train[df_train.columns[-1]].to_numpy()
        y_hat_train = U5.predict_logistic(X_train, coeffs)
        lc_acc_2d_train.set(float(np.mean(y_hat_train == y_train)))

    
    @output
    @render_plotly
    def lc_plot_2d():
        df = lc_df_2d.get()
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="Upload a CSV to begin",
                x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            return fig

        x_col = df.columns[0]
        y_col = df.columns[1]
        
        # colors: red and blue for classes 0 and 1
        hue_col = df.columns[2]
        colors = ['red' if val == 0 else 'blue' for val in df[hue_col]]

        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=colors,
            title="Ground truth (classes)"
        )
        fig.update_layout(template="plotly_white")
        return fig


    @output
    @render_plotly
    def lc_plot_prect_2d():
        df = lc_df_2d.get()
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="Upload a CSV to begin",
                x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            # no background color
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")
            return fig
        
        coeffs = lc_coeffs_2d.get()
        fig = go.Figure()
        if coeffs is None:
            fig.add_annotation(
                text="Train the classifier to see predictions",
                x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            return fig
        

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Predictions on training set", "Predictions on test set")
        )

        

        

        dataset_train = lc_df_2d_train.get()
        dataset_test = lc_df_2d_test.get()

        x_col = df.columns[0]
        y_col = df.columns[1]

        # Use a 2-color palette similar to px default
        palette = px.colors.qualitative.Plotly
        class_colors = {
            0: palette[0],
            1: palette[1],
        }

        # ---------- TRAIN ----------
        if not dataset_train.empty:
            preds_train = U5.predict_logistic(dataset_train.drop(columns="y"), coeffs)
            unique_classes = np.unique(preds_train)

            for idx_cls, cls in enumerate(unique_classes):
                mask = preds_train == cls
                color = class_colors.get(int(cls), palette[idx_cls % len(palette)])

                fig.add_trace(
                    go.Scatter(
                        x=dataset_train.loc[mask, x_col],
                        y=dataset_train.loc[mask, y_col],
                        mode="markers",
                        marker=dict(color=color),
                        name=f"Class {cls} (train)",
                        showlegend=True,
                    ),
                    row=1, col=1
                )

        # ---------- TEST ----------
        if not dataset_test.empty:
            preds_test = U5.predict_logistic(dataset_test.drop(columns="y"), coeffs)
            unique_classes = np.unique(preds_test)

            for idx_cls, cls in enumerate(unique_classes):
                mask = preds_test == cls
                color = class_colors.get(int(cls), palette[idx_cls % len(palette)])

                # avoid duplicate legend labels: only show in train subplot
                fig.add_trace(
                    go.Scatter(
                        x=dataset_test.loc[mask, x_col],
                        y=dataset_test.loc[mask, y_col],
                        mode="markers",
                        marker=dict(color=color),
                        name=f"Class {cls} (test)",
                        showlegend=True,
                    ),
                    row=1, col=2
                )

        fig.update_xaxes(title_text=x_col, row=1, col=1)
        fig.update_yaxes(title_text=y_col, row=1, col=1)
        fig.update_xaxes(title_text=x_col, row=1, col=2)
        fig.update_yaxes(title_text=y_col, row=1, col=2)

        fig.update_layout(
            template="plotly_white",
            legend_title_text="Predicted class",
        )
        return fig


    
    @output
    @render.text
    def lc_coeffs_txt_2d():
        coeffs = lc_coeffs_2d.get()
        if coeffs is None:
            return "(No coefficients yet — click 'Train classifier')"
        return json.dumps({"coefficients": coeffs}, indent=2)
    
    @output
    @render.text
    def lc_metrics_2d():
        acc = lc_acc_2d_test.get()
        acc_train = lc_acc_2d_train.get()
        if acc is None and acc_train is None:
            return ""
        return f"Test accuracy: {acc*100:.1f}%, Train accuracy: {acc_train*100:.1f}%"




    # -------------------- MNIST / Fashion — Linear Classifier ----------------

    mn_coeffs = reactive.Value(None)
    mn_accuracy = reactive.Value(None)
    mn_samples = reactive.Value(None)  # (images, true, pred)
    mn_cm_mat = reactive.Value(None)

    mn_progress_pct = reactive.Value(0)
    mn_progress_msg = reactive.Value("")

    mn_mis = reactive.Value(None)

    @output
    @render.ui
    def mn_progress():
        pct = int(mn_progress_pct())
        msg = mn_progress_msg()
        return ui.div(
            ui.div(
                {"class": "progress", "style": "height: 1.2rem;"},
                ui.div({
                    "class": "progress-bar",
                    "role": "progressbar",
                    "style": f"width: {pct}%;",
                    "aria-valuenow": str(pct),
                    "aria-valuemin": "0",
                    "aria-valuemax": "100",
                }, f"{pct}%")
            ),
            ui.tags.small(msg),
        )


    def _get_loaders():
        batch = int(input.mn_batch())
        seed = int(input.mn_seed())
        if input.mn_ds() == "mnist":
            train, test = U5.get_dataset_mnist(
                batch_size=batch,
                horizontal_flip_p=float(input.mn_hflip()),
                invert=bool(input.mn_invert()),
                seed=seed,
            )
        else:
            train, test = U5.get_dataset_fashionmnist(
                batch_size=batch,
                horizontal_flip_p=float(input.mn_hflip()),
                invert=bool(input.mn_invert()),
                seed=seed,
            )
        return train, test

    @reactive.effect
    @reactive.event(input.mn_train)
    async def _mn_train():
        U5.set_seed(int(input.mn_seed()))

        # Reset progress
        mn_progress_pct.set(0)
        mn_progress_msg.set("Initializing…")
        mn_coeffs.set(None)
        await reactive.flush()

        train_loader, _ = _get_loaders()

        # Build a plain logistic/linear classifier (Flatten -> Linear(784->10))
        # Works for both MNIST and Fashion-MNIST (28x28, 10 classes)
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 10)
        )
        device = torch.device("cpu") # cpu is good enough for this
        model.to(device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=float(input.mn_lr()),
                                    momentum=float(input.mn_mom()))

        epochs = int(input.mn_epochs())
        n_batches = max(1, len(train_loader))

        for ep in range(epochs):
            model.train(True)
            for i, (xb, yb) in enumerate(train_loader):
                xb = xb.to(device)
                yb = yb.to(device)

                logits = model(xb)
                loss = loss_fn(logits, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update progress per batch
                done = ep * n_batches + (i + 1)
                total = epochs * n_batches
                pct = int(100 * done / max(1, total))
                mn_progress_pct.set(pct)
                mn_progress_msg.set(f"Epoch {ep+1}/{epochs} · batch {i+1}/{n_batches}")
                await reactive.flush()

        # Export weights in the SAME format that U5.minimize_ce() returns
        # => tuple((bias, *weights) for each class)
        b = model[1].bias.detach().cpu().numpy()
        W = model[1].weight.detach().cpu().numpy()   # shape (10, 784)
        coeffs = tuple((float(bi), *Wi.tolist()) for bi, Wi in zip(b, W))
        mn_coeffs.set(coeffs)

        mn_progress_msg.set("Training complete")
        mn_progress_pct.set(100)
        await reactive.flush()


    @reactive.effect
    @reactive.event(input.mn_eval)
    def _mn_eval():
        coeffs = mn_coeffs.get()
        if coeffs is None:
            return
            
        U5.set_seed(int(input.mn_seed()))
        _, test = _get_loaders()

        # Unpack coefficients into bias + weight matrix
        coeffs_list = list(coeffs)
        b = np.array([c[0] for c in coeffs_list])        # (K,)
        W = np.array([c[1:] for c in coeffs_list])       # (K, D)

        all_preds = []
        all_targets = []

        shown_imgs = []
        shown_true = []
        shown_pred = []

        mis_imgs = []
        mis_true = []
        mis_pred = []
        mis_probs = []

        for xb, yb in test:
            # xb: (batch, 1, 28, 28), yb: (batch,)
            x_np = xb.view(xb.shape[0], -1).numpy()      # (batch, D)
            y_np = yb.numpy()

            # Softmax using the trained linear classifier
            logits = x_np @ W.T + b                      # (batch, K)
            logits -= logits.max(axis=1, keepdims=True)  # stability
            exp_scores = np.exp(logits)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)  # (batch, K)

            preds = np.argmax(probs, axis=1)

            all_preds.append(preds)
            all_targets.append(y_np)

            # Collect up to 9 examples for the existing mn_examples plot
            if len(shown_imgs) < 9:
                take = min(9 - len(shown_imgs), xb.shape[0])
                shown_imgs.append(xb[:take].numpy())
                shown_true.append(y_np[:take])
                shown_pred.append(preds[:take])

            # Collect up to 10 misclassified examples for the table
            if len(mis_imgs) < 10:
                mis_mask = preds != y_np
                mis_indices = np.where(mis_mask)[0]
                for idx in mis_indices:
                    mis_imgs.append(xb[idx].numpy())         # (1, 28, 28)
                    mis_true.append(int(y_np[idx]))
                    mis_pred.append(int(preds[idx]))
                    mis_probs.append(probs[idx].copy())
                    if len(mis_imgs) >= 10:
                        break

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        acc = float(np.mean(all_preds == all_targets))
        mn_accuracy.set(acc)

        # Existing 3x3 sample grid
        if shown_imgs:
            imgs = np.concatenate(shown_imgs, axis=0)
            tr = np.concatenate(shown_true, axis=0)
            pr = np.concatenate(shown_pred, axis=0)
            mn_samples.set((imgs[:9], tr[:9], pr[:9]))

        # Store misclassified examples (if any)
        if mis_imgs:
            imgs_arr = np.stack(mis_imgs, axis=0)             # (n, 1, 28, 28)
            true_arr = np.array(mis_true)
            pred_arr = np.array(mis_pred)
            probs_arr = np.stack(mis_probs, axis=0)           # (n, K)
            mn_mis.set((imgs_arr, true_arr, pred_arr, probs_arr))
        else:
            mn_mis.set(None)

        # Confusion matrix
        try:
            from sklearn.metrics import confusion_matrix
            mn_cm_mat.set(confusion_matrix(all_targets, all_preds, labels=list(range(10))))
        except Exception:
            mn_cm_mat.set(None)

    @output
    @render.ui
    def mn_mis_table():
        data = mn_mis.get()
        if data is None:
            return ui.p("No misclassifications stored yet. Train and evaluate the model first.")

        imgs, y_true, y_pred, probs = data
        import base64
        from io import BytesIO

        # --- Label names depending on dataset ---
        if input.mn_ds() == "mnist":
            class_names = [str(i) for i in range(10)]
        else:
            class_names = [
                "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
            ]

        num_rows = imgs.shape[0]
        num_classes = probs.shape[1]

        # ---- TABLE HEADER ----
        header = ui.tags.tr(
            ui.tags.th("Image"),
            ui.tags.th("True label"),
            ui.tags.th("Predicted"),
            *[ui.tags.th(name) for name in class_names]
        )

        rows = []

        # ---- TABLE BODY ----
        for i in range(num_rows):
            img_2d = imgs[i, 0]  # (28, 28)
            true_idx = int(y_true[i])
            pred_idx = int(y_pred[i])

            # Convert image to base64
            buf = BytesIO()
            plt.imsave(buf, img_2d, cmap="gray")
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode("ascii")
            img_tag = ui.tags.img(
                src=f"data:image/png;base64,{b64}",
                style="width:48px; height:48px;"
            )

            # Probability cells with highlighting:
            #  - predicted class: table-primary
            #  - true class:      table-success
            prob_cells = []
            for k in range(num_classes):
                cell_text = f"{probs[i, k]:.3f}"
                classes = []
                if k == pred_idx:
                    classes.append("table-primary")
                if k == true_idx:
                    classes.append("table-success")
                attrs = {"class": " ".join(classes)} if classes else {}
                prob_cells.append(
                    ui.tags.td(attrs, cell_text)
                )

            row = ui.tags.tr(
                ui.tags.td(img_tag),
                ui.tags.td({"class": "table-success"}, class_names[true_idx]),
                ui.tags.td({"class": "table-primary"}, class_names[pred_idx]),
                *prob_cells
            )
            rows.append(row)

        # Small legend explaining colors
        legend = ui.div(
            {"class": "mb-2"},
            ui.tags.span({"class": "badge bg-primary me-2"}, "Predicted class"),
            ui.tags.span({"class": "badge bg-success"}, "True class"),
        )

        table = ui.tags.table(
            {"class": "table table-sm table-bordered align-middle"},
            ui.tags.thead(header),
            ui.tags.tbody(*rows)
        )

        return ui.div(legend, table)


    @output
    @render.text
    def mn_acc():
        acc = mn_accuracy.get()
        if acc is None:
            return ""
        name = "MNIST" if input.mn_ds() == "mnist" else "Fashion-MNIST"
        return f"Test accuracy on {name}: {acc*100:.2f}%"

    @output
    @render.plot
    def mn_examples():
        data = mn_samples.get()
        fig, ax = plt.subplots()

        if data is None:
            ax.text(0.5, 0.5, "Click 'Train' and then 'Evaluate'", ha="center", va="center")
            ax.axis("off")
            return fig

        imgs, y_true, y_pred = data

        # ----- Label names depending on dataset -----
        if input.mn_ds() == "mnist":
            class_names = [str(i) for i in range(10)]
        else:
            class_names = [
                "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
            ]

        fig, axes = plt.subplots(3, 3, figsize=(6, 6))
        idx = 0
        for r in range(3):
            for c in range(3):
                axes[r, c].imshow(imgs[idx, 0], cmap="gray")

                true_idx = int(y_true[idx])
                pred_idx = int(y_pred[idx])
                true_label = class_names[true_idx]
                pred_label = class_names[pred_idx]

                axes[r, c].set_title(f"pred {pred_label} / true {true_label}", fontsize=8)
                axes[r, c].axis("off")
                idx += 1

        plt.tight_layout()
        return fig

    
    
    @output
    @render_plotly
    def mn_cm():
        cm = mn_cm_mat.get()
        fig = go.Figure()

        if cm is None:
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            return fig

        labels = list(range(cm.shape[0]))

        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                colorscale="Blues",
                showscale=True,
                text=cm,
                texttemplate="%{text}",
                textfont=dict(size=10)
            )
        )
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="True",
            template="plotly_white"
        )
        fig.update_yaxes(autorange="reversed")  # to match matrix-style layout
        return fig


    @render.download(filename=lambda: "u5_mnist_coeffs.json")
    def mn_dl_coeffs():
        coeffs = mn_coeffs.get() or []
        yield json.dumps({"coefficients": coeffs}, indent=2)

    @output
    @render.plot
    def mnist_samples():
        # Load MNIST once
        train_loader, _ = U5.get_dataset_mnist(
            batch_size=9,
            horizontal_flip_p=0.0,
            invert=False,
            seed=42,
        )

        # Get first batch (9 images)
        imgs, labels = next(iter(train_loader))
        imgs = imgs.numpy()

        fig, axes = plt.subplots(3, 3, figsize=(6, 6))
        idx = 0
        for r in range(3):
            for c in range(3):
                axes[r, c].imshow(imgs[idx, 0], cmap="gray")
                axes[r, c].set_title(f"Digit: {int(labels[idx])}")
                axes[r, c].axis("off")
                idx += 1

        return fig


    @output
    @render.plot
    def fashionmnist_samples():
        train_loader, _ = U5.get_dataset_fashionmnist(
            batch_size=9,
            horizontal_flip_p=0.0,
            invert=False,
            seed=42,
        )

        imgs, labels = next(iter(train_loader))
        imgs = imgs.numpy()

        # Class names for display
        fashion_labels = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]

        fig, axes = plt.subplots(3, 3, figsize=(6, 6))
        idx = 0
        for r in range(3):
            for c in range(3):
                axes[r, c].imshow(imgs[idx, 0], cmap="gray")
                axes[r, c].set_title(fashion_labels[int(labels[idx])])
                axes[r, c].axis("off")
                idx += 1

        return fig

    @reactive.effect
    @reactive.event(input.pr_choice)
    def _pr_reset_on_choice_change():
        # Clear data and fitted model when switching between Custom/Random/Classes
        pr_df.set(pd.DataFrame())
        pr_coeffs.set(None)
        pr_true_coefficients.set(None)

    @reactive.effect
    @reactive.event(input.mn_ds)
    def _mn_reset_on_dataset_change():
        # Clear everything derived from the old dataset
        mn_coeffs.set(None)
        mn_accuracy.set(None)
        mn_samples.set(None)
        mn_cm_mat.set(None)
        mn_mis.set(None)

        # Also reset the progress bar
        mn_progress_pct.set(0)
        mn_progress_msg.set("")

    # --- Reset 1D logistic when settings change -----------------------------

    # Data-related settings: changing these conceptually changes the dataset
    @reactive.effect
    @reactive.event(input.lc_seed, input.lc_n)
    def _lc_reset_data_on_change():
        # Clear data + model so plot goes back to "Click 'Generate data'"
        lc_df.set(pd.DataFrame())
        lc_coeffs.set(None)
        lc_acc.set(None)

    # Training-related settings: keep data, but clear the learned model
    @reactive.effect
    @reactive.event(input.lc_epochs, input.lc_lr, input.lc_mom)
    def _lc_reset_model_on_change():
        lc_coeffs.set(None)
        lc_acc.set(None)

    # --- Reset 2D logistic when settings change -----------------------------

    # Changing data-related settings: new seed, header handling, or a new file
    @reactive.effect
    @reactive.event(input.lc_seed_2d, input.lc_ignore_header_2d, input.lc_file_2d)
    def _lc2d_reset_on_data_change():
        lc_coeffs_2d.set(None)
        lc_acc_2d_train.set(None)
        lc_acc_2d_test.set(None)
        lc_df_2d_train.set(pd.DataFrame())
        lc_df_2d_test.set(pd.DataFrame())
        # Keep lc_df_2d so the "Ground truth" scatter still shows

    # Changing training / split settings: model / split no longer valid
    @reactive.effect
    @reactive.event(input.lc_split_2d, input.lc_epochs_2d, input.lc_lr_2d, input.lc_mom_2d)
    def _lc2d_reset_on_training_change():
        lc_coeffs_2d.set(None)
        lc_acc_2d_train.set(None)
        lc_acc_2d_test.set(None)
        lc_df_2d_train.set(pd.DataFrame())
        lc_df_2d_test.set(pd.DataFrame())


app = App(app_ui, server)


from pathlib import Path
from shiny import App, ui, render, reactive
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets as skds
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn import datasets
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shinywidgets import render_plotly

def load_wine():
    wine_data = datasets.load_wine()
    data = pd.DataFrame(wine_data['data'], columns=wine_data['feature_names']) # type: ignore
    data['cultivator'] = wine_data['target'] # type: ignore
    return data

def load_penguins():
    penguins = sns.load_dataset("penguins").dropna()
    penguins = penguins.drop(["island", "sex"], axis=1)
    penguins["species"] = penguins["species"].astype("category").cat.codes
    penguins.index = range(len(penguins)) # type: ignore
    return penguins

def load_iris_df():
    ds = skds.load_iris()
    X = pd.DataFrame(ds.data, columns=ds.feature_names)
    y = pd.Series(ds.target, name="species")
    return pd.concat([X, y], axis=1)

def load_breast_cancer_df():
    ds = skds.load_breast_cancer()
    X = pd.DataFrame(ds.data, columns=ds.feature_names)
    y = pd.Series(ds.target, name="diagnosis")
    return pd.concat([X, y], axis=1)



DATASETS = {
    "wine":        ("Wine Dataset",         load_wine),
    "penguins":    ("Penguins Dataset",     load_penguins),
    "iris":        ("Iris Dataset",         load_iris_df),
    "breast":      ("Breast Cancer Dataset",load_breast_cancer_df),
}

def class_labels(dataset_name):
    if dataset_name == "wine":
        return {0: "Cultivator 0", 1: "Cultivator 1", 2: "Cultivator 2"}
    elif dataset_name == "penguins":
        return {0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}
    elif dataset_name == "iris":
        return {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    elif dataset_name == "breast":
        return {0: "Malignant", 1: "Benign"}
    else:
        return {}
    


app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.link(
            rel="stylesheet",
            href="vendor/zephyr/bootstrap.min.css"
        ),
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
    ),
    ui.h2("Hands-on AI I - Unit 1: Tabular Data", class_="text-center mb-4"),
    ui.navset_tab(
        ui.nav_panel("Intro",
            ui.row(
                ui.column(12,
                    ui.p(),
                    ui.p("This application demonstrates various data analysis \
                    techniques using several well-known datasets. You can explore dataset \
                    statistics, visualize data, perform dimensionality reduction, \
                    and conduct a correlation analysis."),
                    ui.p("Important note: make sure to use the supplied Anaconda \
                    environment to run this application."),
                    
                    ui.h4("Wine Data"),
                    ui.p("The Wine dataset is composed of various measurements of \
                    different wine attributes (e.g. the alcohol concentration). \
                    The data set distinguishes three different classes, one for each \
                    cultivator. It was published/donored by S. Aeberhard and originally \
                    gathered by:"),
                    ui.markdown("""
                    _Forina, M. et al, PARVUS - An Extendible Package for Data Exploration, Classification and Correlation. Institute of Pharmaceutical and Food Analysis and Technologies, Via Brigata Salerno, 16147 Genoa, Italy._
                    """),
                    ui.p("Currently, it is maintained by the UCI Machine Learning Repository:"),
                    ui.markdown("""
                    _Lichman, M. (2013). [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php). Irvine, CA: University of California, School of Information and Computer Science._
                    """),
                    ui.p("The wine dataset is a table with all 178 samples (we start to \
                    count at 0). Tabular data can have columns in various data types. In \
                    our case the thirteen features are given in floating point numbers \
                    (recall primitive data types). We would refer to the cultivator as a \
                    ", ui.strong("label "), "or ", ui.strong("target "), "rather than a \
                    feature, because we want to predict the cultivator using all other features."),
                    ui.p("Summarizing, the popular wine data set contains results of a \
                    chemical analysis for \(n=178\) different wines from three different \
                    classes, namely:"),
                    ui.tags.ul(
                        ui.tags.li("Cultivator 0: \(n_{c0}=59\)"),
                        ui.tags.li("Cultivator 1: \(n_{c1}=71\)"),
                        ui.tags.li("Cultivator 2: \(n_{c2}=48\)")
                    ),
                    ui.p("Moreover, we have the following \(d=13\) features:"),
                    ui.tags.ul(
                        ui.tags.li("Alcohol"),
                        ui.tags.li("Malic acid"),
                        ui.tags.li("Ash"),
                        ui.tags.li("Alcalinity of ash"),
                        ui.tags.li("Magnesium"),
                        ui.tags.li("Total phenols"),
                        ui.tags.li("Flavanoids"),
                        ui.tags.li("Nonflavanoid phenols"),
                        ui.tags.li("Proanthocyanins"),
                        ui.tags.li("Color intensity"),
                        ui.tags.li("Hue"),
                        ui.tags.li("OD280/OD315 of diluted wines"),
                        ui.tags.li("Proline")
                    ),
                    ui.h4("Penguins Data"),
                    ui.p("The Penguins dataset contains 344 samples with measurements of penguin species \
                    from three different islands in Antarctic."),
                    ui.p("The target variable is 'species', which indicates the species of the penguin:"),
                    ui.tags.ul(
                        ui.tags.li("Adelie: \(n_{a}=152\)"),
                        ui.tags.li("Chinstrap: \(n_{c}=73\)"),
                        ui.tags.li("Gentoo: \(n_{g}=119\)")
                    ),
                    ui.p("Given the following \(d=4\) features:"),
                    ui.tags.ul(
                        ui.tags.li("Bill length (mm)"),
                        ui.tags.li("Bill depth (mm)"),
                        ui.tags.li("Flipper length (mm)"),
                        ui.tags.li("Body mass (g)")
                    ),
                    ui.p("The dataset is available from the seaborn library and was \
                    originally published by:"),
                    ui.markdown("""
                    _Palmer, T. (2020). [Palmer Penguins: An Introduction to Data Exploration](https://allisonhorst.github.io/palmerpenguins/). Retrieved from https://allisonhorst.github.io/palmerpenguins/_.
                    """),
                    ui.h4("Iris Data"),
                    ui.p("The Iris dataset is a classic dataset in machine learning \
                    and statistics. It contains measurements for 150 samples of iris flowers from three \
                    different species: Setosa, Versicolor, and Virginica. The dataset \
                    includes features such as sepal length, sepal width, petal length, \
                    and petal width."),
                    ui.p("The target variable is 'species', which indicates the species of the iris flower:"),
                    ui.tags.ul(
                        ui.tags.li("Setosa: \(n_{s}=50\)"),
                        ui.tags.li("Versicolor: \(n_{v}=50\)"),
                        ui.tags.li("Virginica: \(n_{v}=50\)")
                    ),
                    ui.p("Given the following \(d=4\) features:"),
                    ui.tags.ul(
                        ui.tags.li("Sepal length (cm)"),
                        ui.tags.li("Sepal width (cm)"),
                        ui.tags.li("Petal length (cm)"),
                        ui.tags.li("Petal width (cm)")
                    ),
                    ui.p("The dataset is available from the sklearn library and was \
                    originally published by:"),
                    ui.markdown("""
                    _Fisher, R.A. (1936). The use of multiple measurements in taxonomic problems. Annual Eugenics, 7(2), 179-188. doi:10.1111/j.1469-1809.1936.tb02137.x_.
                    """),
                    ui.h4("Breast Cancer Data"),
                    ui.p("The Breast Cancer dataset contains \(n=569\) samples with measurements of various \
                    features related to breast cancer tumors. The dataset includes features \
                    such as mean radius, mean texture, mean perimeter, and diagnosis (malignant \
                    or benign). The target variable is 'diagnosis', which indicates whether the tumor is malignant or benign:"),
                    ui.tags.ul(
                        ui.tags.li("Malignant: \(n_{m}=212\)"),
                        ui.tags.li("Benign: \(n_{b}=357\)")
                    ),
                    ui.p("Given the following \(d=30\) features (list truncated):"),
                    ui.tags.ul(
                        ui.tags.li("Mean radius"),
                        ui.tags.li("Mean texture"),
                        ui.tags.li("Mean perimeter"),
                        ui.tags.li("Mean area"),
                        ui.tags.li("Mean smoothness"),
                        ui.tags.li("Mean compactness"),
                        ui.tags.li("Mean concavity"),
                        ui.tags.li("Mean concave points"),
                        ui.tags.li("Mean symmetry"),
                        ui.tags.li("Mean fractal dimension"),
                    ),
                    ui.p("The dataset is available from the sklearn library and was \
                    originally published by:"),
                    ui.markdown("""
                    _Wolberg, W.H., Street, W.N., & Mangasarian, O.L. (1992). \
                    [Breast cancer Wisconsin (diagnostic) dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)). UCI Machine Learning Repository._
                    """)
                )
            )
        ),
        # Dataset Overview Tab
        ui.nav_panel("Dataset Overview",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.tags.aside(
						ui.h4("Dataset Selection"),
                        ui.input_select(
                            "dataset", "Choose a Dataset:",
                            choices={
                                "wine": "Wine",
                                "penguins": "Penguins",
                                "iris": "Iris",
                                "breast": "Breast Cancer (Wisconsin)",
                            },
                            selected="wine",
                        ),
                        ui.br(),
                        ui.input_action_button("load_data", "Load Data", class_="btn-primary"),
                        class_="sidebar"
                    ), width=300
                ),
                ui.panel_conditional("input.load_data > 0",
                    ui.h4("Dataset Information"),
                    ui.output_text_verbatim("dataset_info"),
                    ui.br(),
                    ui.h4("Dataset"),
                    ui.output_data_frame("data_table"),
                    ui.br(),
                    ui.h4("Column Info"),
                    ui.output_text_verbatim("dataset_full_info"),
                    ui.br(),
                    ui.h4("Statistical Summary"),
                    ui.output_data_frame("data_summary"),
                    ui.br(),
                    ui.h4("Value Counts of Target Column"),
                    ui.output_text_verbatim("target_valuecount")
                )
            )
        ),
        ui.nav_panel("Data Visualization",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.tags.aside(
                        ui.h4("Visualization Controls"),
                        ui.input_selectize(
                            "viz_features", "Select Features:",
                            choices=[], multiple=True
                        ),
                        ui.input_select(
                            "viz_type", "Visualization Type:",
                            choices={
                                "pairplot": "Pairwise Relationships",
                                "boxplot": "Box Plots",
                                "histogram": "Histograms",
                                "scatter": "Scatter Plot",
                                "violinplot": "Violin Plots"
                            }
                        ),
                        ui.input_checkbox("color_by_target", "Color by Target", value=True),
                        ui.input_action_button("generate_viz", "Generate Visualization", class_="btn-primary"),
                        class_="sidebar"
                    ), width=300
                ),
                ui.div(
                    ui.output_plot("visualization_plot", height="600px"),
                    class_="plot-container"
                )
            )
        ),
        ui.nav_panel("Dimensionality Reduction",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.tags.aside(
                        ui.h4("Reduction Parameters"),
                        ui.input_select(
                            "reduction_method", "Method:",
                            choices={"pca": "PCA", "tsne": "t-SNE"}
                        ),
                        ui.input_numeric("n_components", "Number of Components:", value=2, min=2, max=3),
                        ui.input_checkbox("standardize", "Standardize Data", value=True),
                        ui.help_text("Standardizing data to have mean=0 and variance=1 is often beneficial for PCA and t-SNE."),
                        ui.panel_conditional(
                            "input.reduction_method === 'tsne'",
                            ui.input_slider("perplexity", "Perplexity:", min=5, max=50, value=30)
                        ),
                        ui.hr(),
                        ui.input_action_button("apply_reduction", "Apply Reduction", class_="btn-primary"),
                        ui.br(), ui.br(),
                        
                        class_="sidebar"
                    ), width=300
                ),
                ui.column(12,
					ui.p(),
                    ui.div(
                        ui.output_plot("reduction_plot", height="600px"),
                        class_="plot-container"
                    ),
                    ui.panel_conditional("input.reduction_method === 'pca'",
                        ui.output_text_verbatim("explained_variance",)
                    ),
                )
            )
        ),
        ui.nav_panel("Correlation Analysis",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.tags.aside(
                        ui.h4("Features"),
                        ui.input_selectize(
                            "analysis_features", "Select Features for Analysis:",
                            choices=[], multiple=True
                        ),
                        ui.input_action_button("run_analysis", "Run Analysis", class_="btn-primary"),
                        class_="sidebar"
                    ), width=300
                ),
                ui.column(12,
					ui.p(),
					ui.output_text_verbatim("analysis_results"),
                    ui.div(
                        ui.output_plot("analysis_plot", height="600px"),
                        class_="plot-container"
                    )
                )
            )
        )
    )
)


def server(input, output, session):

    @reactive.calc
    def get_data():
        if input.dataset() == "wine":
            return load_wine()
        elif input.dataset() == "iris":
            return load_iris_df()
        elif input.dataset() == "breast":
            return load_breast_cancer_df()
        else:
            return load_penguins()
    
    @reactive.calc
    def get_features():
        data = get_data()
        target_col = 'cultivator' if input.dataset() == 'wine' else 'species'
        return [col for col in data.columns if col != target_col]
    
    @reactive.calc
    def get_target_column():
        if input.dataset() == "wine":
            return 'cultivator'
        elif input.dataset() == "iris":
            return 'species'
        elif input.dataset() == "breast":
            return 'diagnosis'
        else:
            return 'species'
    
    # Update feature choices when dataset changes
    @reactive.effect
    def update_feature_choices():
        features = get_features()
        ui.update_selectize("viz_features", choices=features, selected=features[:3])
        ui.update_selectize("analysis_features", choices=features, selected=features[:4])
    
    @output
    @render.text
    @reactive.event(input.load_data)
    def dataset_info():
        data = get_data()
        labels_dict = class_labels(input.dataset())
        labels = "\n".join([f"{k}: {v}" for k, v in labels_dict.items()])
        return f"Shape: {data.shape}\nFeatures: {len(get_features())}\nSamples: {len(data)}\n\nClass labels:\n{labels}\n"

    
    @output
    @render.text
    @reactive.event(input.load_data)
    def dataset_full_info():
        buf = io.StringIO()
        data_fullinfo = get_data().info(buf=buf)
        s = buf.getvalue()
        return s
    
    @output
    @render.data_frame
    @reactive.event(input.load_data)
    def data_table():
        return get_data()
    
    @output
    @render.text
    @reactive.event(input.load_data)
    def target_valuecount():
        data = get_data()
        target_col = get_target_column()
        value_counts = data[target_col].value_counts()
        return value_counts
    
    @output
    @render.data_frame
    @reactive.event(input.load_data)
    def data_summary():
        data = get_data()
        target_col = get_target_column()
        summary = data.drop(columns=[target_col]).describe().round(3).rename_axis("stat").reset_index()
        return summary
    
    @output
    @render.plot
    @reactive.event(input.generate_viz)
    def visualization_plot():
        data = get_data()
        features = input.viz_features()
        target_col = get_target_column() if input.color_by_target() else None
        
        if not features:
            return plt.figure()
        
        fig, axes = plt.subplots(figsize=(12, 8))
        
        if input.viz_type() == "pairplot":
            if len(features) >= 2:
                g = sns.pairplot(data=data, vars=features[:4], hue=target_col, palette="deep")
                return g.fig
        
        elif input.viz_type() == "boxplot":
            fig, axes = plt.subplots(1, len(features), figsize=(15, 6))
            if len(features) == 1:
                axes = [axes]
            for i, feature in enumerate(features):
                if target_col:
                    sns.boxplot(data=data, x=target_col, y=feature, ax=axes[i], palette="deep")
                else:
                    sns.boxplot(data=data, y=feature, ax=axes[i])
                axes[i].set_title(feature)
            for ax in axes:
                ax.set_ylabel('')
        
        elif input.viz_type() == "histogram":
            fig, axes = plt.subplots(1, len(features), figsize=(15, 6))
            if len(features) == 1:
                axes = [axes]
            for i, feature in enumerate(features):
                sns.histplot(data=data, x=feature, hue=target_col, kde=True, ax=axes[i])
                axes[i].set_title(feature)
            for ax in axes:
                ax.set_ylabel('')
            
        elif input.viz_type() == "violinplot":
            fig, axes = plt.subplots(1, len(features), figsize=(15, 6))
            if len(features) == 1:
                axes = [axes]
            for i, feature in enumerate(features):
                sns.violinplot(data=data, x=target_col, y=feature, ax=axes[i], palette="deep")
                axes[i].set_title(feature)
            for ax in axes:
                ax.set_ylabel('')
        
        elif input.viz_type() == "scatter" and len(features) >= 2:
            sns.scatterplot(data=data, x=features[0], y=features[1], 
                          hue=target_col, palette="deep", s=60)
            plt.title(f"{features[0]} vs {features[1]}")
        return fig
    
    @output
    @render.plot
    @reactive.event(input.apply_reduction)
    def reduction_plot():
        data = get_data()
        target_col = get_target_column()
        features_data = data.drop(columns=[target_col])
        
        if input.standardize():
            features_data = pd.DataFrame(scale(features_data), 
                                       columns=features_data.columns, 
                                       index=features_data.index)      
        if input.reduction_method() == "pca":
            reducer = PCA(n_components=input.n_components())
            reduced_data = reducer.fit_transform(features_data)
            loadings = pd.DataFrame(
                reducer.components_.T,
                index=features_data.columns,
                columns=[f"PC{i+1}" for i in range(reducer.n_components_)]
            ).sort_values("PC1", key=np.abs, ascending=False)
        else:
            reducer = TSNE(n_components=input.n_components(), 
                          perplexity=input.perplexity(), 
                          random_state=42)
            reduced_data = reducer.fit_transform(features_data)
        fig = plt.figure(figsize=(10, 8))
        if input.n_components() == 2:
            plt.scatter(reduced_data[:, 0], reduced_data[:, 1], 
                       c=data[target_col], cmap='viridis', s=60)
            plt.xlabel(f"Component 1")
            plt.ylabel(f"Component 2")
        else:
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2],
                               c=data[target_col], cmap='viridis', s=60)
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_zlabel("Component 3")
        labels_dict = class_labels(input.dataset())
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=labels_dict.get(i, str(i)),
                              markerfacecolor=plt.cm.viridis(i / max(labels_dict.keys())), markersize=10) 
                   for i in labels_dict.keys()]
        plt.legend(title=target_col, handles=handles)
        plt.title(f"{input.reduction_method().upper()} - {input.n_components()}D Projection")
        return fig
    
    @output
    @render.data_frame
    def pca_loadings():
        if input.reduction_method() != "pca":
            return pd.DataFrame({"info":["Available for PCA only"]})
        data = get_data()
        target_col = get_target_column()
        X = data.drop(columns=[target_col]).round(3)
        if input.standardize():
            X = pd.DataFrame(scale(X), columns=X.columns, index=X.index)
        pca = PCA(n_components=min(input.n_components(), len(X.columns)))
        pca.fit(X)
        return pd.DataFrame(
            pca.components_.T,
            index=X.columns,
            columns=[f"PC{i+1}" for i in range(pca.n_components_)]
    )

    @output
    @render.text
    @reactive.event(input.apply_reduction)
    def explained_variance():
        if input.reduction_method() == "pca":
            data = get_data()
            target_col = get_target_column()
            features_data = data.drop(columns=[target_col])
            
            if input.standardize():
                features_data = pd.DataFrame(scale(features_data))
            
            pca = PCA(n_components=min(input.n_components(), len(features_data.columns)))
            pca.fit(features_data)
            
            variance_text = "Explained Variance Ratio:\n"
            for i, var in enumerate(pca.explained_variance_ratio_):
                variance_text += f"PC{i+1}:   {var:.3f} ({var*100:.1f}%)\n"
            variance_text += f"Total: {sum(pca.explained_variance_ratio_):.3f} ({sum(pca.explained_variance_ratio_)*100:.1f}%)"
            return variance_text
        return "Explained variance only available for PCA"
    
    
    @output
    @render.plot
    @reactive.event(input.run_analysis)
    def analysis_plot():
        data = get_data()
        features = input.analysis_features()
        target_col = get_target_column()
    
        if not features:
            return plt.figure()
    
        fig = plt.figure(figsize=(12, 8))
    
        numeric_data = data.select_dtypes(include=['number'])
        available_features = [f for f in features if f in numeric_data.columns]
        
        if len(available_features) < 2:
            plt.text(0.5, 0.5, 'Not enough numeric features selected for correlation analysis', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title("Correlation Analysis Error")
            return fig
            
        corr_matrix = numeric_data[available_features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title("Feature Correlation Matrix")
    
        plt.tight_layout()
    
        return fig

    @output
    @render.text
    @reactive.event(input.run_analysis)
    def analysis_results():
        data = get_data()
        features = input.analysis_features()
        target_col = get_target_column()
    
        if not features:
            return ""
    
        numeric_data = data.select_dtypes(include=['number'])
        available_features = [f for f in features if f in numeric_data.columns]
    
        if len(available_features) < 2:
            return "Not enough numeric features selected for correlation analysis"
    
        corr_matrix = numeric_data[available_features].corr()
    
        results_text = "Strongest Correlations (Pearson):\n"
        correlations = []
        for i in range(len(available_features)):
            for j in range(i + 1, len(available_features)):
                corr_val = corr_matrix.iloc[i, j]
                correlations.append((available_features[i], available_features[j], abs(corr_val), corr_val))
    
        correlations.sort(key=lambda x: x[2], reverse=True)
    
        for feat1, feat2, abs_corr, corr in correlations[:5]:
            results_text += f"{feat1} - {feat2}: {corr:.3f}\n"

        results_text += "\nWeakest Correlations (Pearson):\n"
        for feat1, feat2, abs_corr, corr in correlations[-5:]:
            results_text += f"{feat1} - {feat2}: {corr:.3f}\n"
    
        return results_text

app = App(app_ui, server, static_assets=Path(__file__).parent / "www")

if __name__ == "__main__":
    app.run()

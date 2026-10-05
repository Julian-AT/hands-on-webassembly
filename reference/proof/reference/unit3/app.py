from __future__ import annotations

from ast import Is
import io
import base64
import os
import textwrap
from pathlib import Path
from tkinter.font import names
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.patches import Ellipse
import plotly.graph_objects as go
from plotly.colors import qualitative as qcolors

from scipy.special import logsumexp
from scipy.spatial import Voronoi, voronoi_plot_2d

from sklearn.datasets import fetch_openml, make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.datasets import make_classification
from sklearn.datasets import load_iris, load_wine, load_breast_cancer

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram as _scipy_dendrogram

from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_plotly

import seaborn as sns
sns.set_theme(style="white")



# -----------------------------------------------------------------------------
# ---- Utilities --------------------------------------------------------------
# -----------------------------------------------------------------------------


def silhouette_plot(data, groups, title=None, xlabel=None, ylabel=None):
    silhouette_avg = silhouette_score(data, groups)
    silhouette_values = silhouette_samples(data, groups)

    unique_labels = np.unique(groups)
    n_clusters = len(unique_labels)
    colors = qcolors.Plotly * (n_clusters // len(qcolors.Plotly) + 1)

    traces = []
    y_lower = 10

    for i, label in enumerate(unique_labels):
        ith_cluster_silhouette_values = silhouette_values[groups == label]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        y_range = np.arange(y_lower, y_upper)
        x_vals = ith_cluster_silhouette_values

        traces.append(go.Scatter(
            x=x_vals,
            y=y_range,
            mode='lines',
            fill='tozerox',
            line=dict(color=colors[i]),
            name=f"Cluster {label}",
            hoverinfo='x+y+name'
        ))

        traces.append(go.Scatter(
            x=[-0.05],
            y=[y_lower + 0.5 * size_cluster_i],
            text=[str(label)],
            mode='text',
            showlegend=False
        ))

        y_lower = y_upper + 10

    # Red dashed line for average silhouette score
    traces.append(go.Scatter(
        x=[silhouette_avg, silhouette_avg],
        y=[0, y_lower],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name=f"Average = {silhouette_avg:.2f}"
    ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title or "Silhouette Plot",
        xaxis_title=xlabel or "Silhouette coefficient values",
        yaxis_title=ylabel or "Cluster",
        yaxis=dict(showticklabels=False),
        plot_bgcolor='white',
        hovermode='closest',
        legend=dict(title="Clusters", bgcolor="rgba(255,255,255,0.7)")
    )
    fig.update_layout(height=600)
    return fig


def dendrogram_plot(linked_clusters,
                    figsize=None,
                    orientation: str = "top",
                    title: str = "Dendrogram",
                    xaxis_title: str = "Samples",
                    yaxis_title: str = "Euclidean distances",
                    labels=None) -> go.Figure:
    """
    Plot a dendrogram with Plotly from a SciPy linkage matrix.

    Parameters
    ----------
    linked_clusters : ndarray
        SciPy linkage matrix (e.g., from scipy.cluster.hierarchy.linkage).
    figsize : tuple[float, float] | None
        (width, height) in inches (roughly). If None, defaults to (5, 5).
    orientation : {"top", "left"}
        "top" = classic vertical dendrogram; "left" = horizontal.
    title, xaxis_title, yaxis_title : str
        Axis/figure titles.
    labels : list[str] | None
        Optional labels for leaves; if None, indices are used.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    ddata = _scipy_dendrogram(
        linked_clusters,
        no_plot=True,
        orientation="top" if orientation == "top" else "right",
        labels=labels
    )

    icoord = ddata["icoord"]
    dcoord = ddata["dcoord"]
    ivl    = ddata["ivl"]

    traces = []
    for xs, ys in zip(icoord, dcoord):
        if orientation == "top":
            traces.append(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(width=1),
                hoverinfo="skip",
                showlegend=False
            ))
        else:
            traces.append(go.Scatter(
                x=ys, y=xs, mode="lines",
                line=dict(width=1),
                hoverinfo="skip",
                showlegend=False
            ))

    fig = go.Figure(traces)

    n_leaves = len(ivl)
    leaf_ticks = [5 + 10*i for i in range(n_leaves)]

    if orientation == "top":
        fig.update_xaxes(
            tickmode="array",
            tickvals=leaf_ticks,
            ticktext=ivl,
            showline=True, linewidth=2, linecolor="black", mirror=True
        )
        fig.update_yaxes(
            title=yaxis_title,
            showline=True, linewidth=2, linecolor="black", mirror=True
        )
        fig.update_layout(xaxis_title=xaxis_title)
    else:
        fig.update_yaxes(
            tickmode="array",
            tickvals=leaf_ticks,
            ticktext=ivl,
            showline=True, linewidth=2, linecolor="black", mirror=True
        )
        fig.update_xaxes(
            title=yaxis_title,
            showline=True, linewidth=2, linecolor="black", mirror=True
        )
        fig.update_layout(yaxis_title=xaxis_title)

    if figsize is None:
        width, height = 5, 5
    else:
        width, height = figsize
    fig.update_layout(
        title=title,
        width=int(width * 100),
        height=int(height * 100),
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode=False,
        showlegend=False
    )
    return fig

#def _lighten_rgba(rgba, amount=0.75):
#    r, g, b, a = rgba
#    return (r + (1-r)*amount, g + (1-g)*amount, b + (1-b)*amount, a)


# --- Decision-region helpers (2-D plot space P) ------------------------------

def _region_grid(P: np.ndarray, pad_ratio: float = 0.05, n: int = 300):
    """Make a 2-D meshgrid that slightly pads the plotted data."""
    xpad = pad_ratio * (P[:, 0].ptp() or 1.0)
    ypad = pad_ratio * (P[:, 1].ptp() or 1.0)
    xs = np.linspace(P[:, 0].min() - xpad, P[:, 0].max() + xpad, n)
    ys = np.linspace(P[:, 1].min() - ypad, P[:, 1].max() + ypad, n)
    XX, YY = np.meshgrid(xs, ys)
    grid = np.column_stack([XX.ravel(), YY.ravel()])
    return xs, ys, XX, YY, grid


# Cached region grid (per data bounds and grid size)
region_cache = reactive.Value({
    "bbox": None, "n": None,
    "xs": None, "ys": None, "XX": None, "YY": None, "grid": None
})


def _get_region_grid(P: np.ndarray, n: int = 300, pad_ratio: float = 0.05):
    xpad = pad_ratio * (P[:, 0].ptp() or 1.0)
    ypad = pad_ratio * (P[:, 1].ptp() or 1.0)
    bbox = (
        float(P[:, 0].min() - xpad), float(P[:, 0].max() + xpad),
        float(P[:, 1].min() - ypad), float(P[:, 1].max() + ypad)
    )
    rc = region_cache()
    if rc["bbox"] == bbox and rc["n"] == n and rc["xs"] is not None:
        return rc["xs"], rc["ys"], rc["XX"], rc["YY"], rc["grid"]

    xs = np.linspace(bbox[0], bbox[1], n)
    ys = np.linspace(bbox[2], bbox[3], n)
    XX, YY = np.meshgrid(xs, ys)
    grid = np.column_stack([XX.ravel(), YY.ravel()])
    region_cache.set({"bbox": bbox, "n": n, "xs": xs, "ys": ys, "XX": XX, "YY": YY, "grid": grid})
    return xs, ys, XX, YY, grid


def _lighten_cmap(base_cmap, k: int, amount: float = 0.85):
    """Return a ListedColormap of lightened colors for region backgrounds."""
    from matplotlib.colors import ListedColormap, to_rgba
    cols = [to_rgba(base_cmap(i)) for i in range(k)]
    light = [(_r + (1 - _r) * amount,
              _g + (1 - _g) * amount,
              _b + (1 - _b) * amount,
              a) for (_r, _g, _b, a) in cols]
    return ListedColormap(light)


def _knn_decision_regions(P: np.ndarray, labels: np.ndarray, XX, YY, grid: np.ndarray,
                          ignore_label: int | None = None, k: int = 1):
    """Compute region map via kNN (default 1-NN) on 2-D plot space."""
    from sklearn.neighbors import KNeighborsClassifier
    L = labels
    if ignore_label is not None:
        mask = (L != ignore_label)
        if not np.any(mask):
            return None
        Xtrain, ytrain = P[mask], L[mask]
    else:
        Xtrain, ytrain = P, L
    if len(np.unique(ytrain)) < 1:
        return None
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(Xtrain, ytrain)
    Z = clf.predict(grid).reshape(XX.shape)
    return Z


def _mog_argmax_region(grid: np.ndarray, means2d: np.ndarray,
                       covs2d: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return argmax component index for each grid point (MAP) in 2-D plot space."""
    k = means2d.shape[0]
    I = np.eye(2)
    eps = 1e-5
    logs = np.empty((k, grid.shape[0]))
    for i in range(k):
        Σ = covs2d[i] + eps * I
        # log N(x | μ, Σ) + log w
        try:
            L = np.linalg.cholesky(Σ)
            logdet = 2.0 * np.sum(np.log(np.diag(L)))
            diff = (grid - means2d[i])    # (n,2)
            sol = np.linalg.solve(L, diff.T)  # (2,n)
            quad = np.sum(sol**2, axis=0)
            logs[i] = -0.5 * (2*np.log(2*np.pi) + logdet + quad) + np.log(max(weights[i], 1e-12))
        except np.linalg.LinAlgError:
            logs[i] = -1e10
    return np.argmax(logs, axis=0)


# --- Data generation/loading functions ---------------------------------------

# --- Artificial data sets ----------------------------------------------------

def data_blobs(n: int = 300, centers: int = 4, seed: int = 42, std: float = 0.60, separation: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_blobs(n_samples=n, centers=centers, cluster_std=std, random_state=seed, center_box=(-separation, separation))
    return X.astype(float), y


def data_moons(n: int = 300, seed: int = 42, noise: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    return X.astype(float), y


def data_circles(n: int = 300, seed: int = 42, noise: float = 0.10, factor: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_circles(n_samples=n, factor=factor, noise=noise, random_state=seed)
    return X.astype(float), y


def data_uniform(n: int = 300, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(low=-5.0, high=5.0, size=(n, 2))
    y = np.zeros(n, dtype=int)
    return X.astype(float), y


def gaussian_blobs(n: int = 600, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    means = np.array([
        [-4.0,  0.0],
        [ 0.0,  4.0],
        [ 4.0,  0.0],
        [ 0.0, -4.0],
    ])

    def rotated_cov(var1: float, var2: float, theta_deg: float) -> np.ndarray:
        th = np.deg2rad(theta_deg)
        R = np.array([[np.cos(th), -np.sin(th)],
                      [np.sin(th),  np.cos(th)]])
        D = np.diag([var1, var2])
        return R @ D @ R.T

    covs = [
        np.array([[0.35, 0.00], [0.00, 0.35]]),
        np.array([[3.00, 0.00], [0.00, 0.25]]),
        np.array([[0.25, 0.00], [0.00, 3.00]]),
        rotated_cov(2.5, 0.15, 45.0),
    ]

    counts = [n // 4] * 4
    for i in range(n - sum(counts)):
        counts[i] += 1

    X_parts, y_parts = [], []
    for k, (mu, Sigma, nk) in enumerate(zip(means, covs, counts)):
        Xk = rng.multivariate_normal(mean=mu, cov=Sigma, size=nk)
        X_parts.append(Xk)
        y_parts.append(np.full(nk, k, dtype=int))

    X = np.vstack(X_parts).astype(float)
    y = np.concatenate(y_parts)

    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def gaussian_blobs2(n = 600, seed = 42) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(n_samples = n, n_features=2, n_redundant=0, random_state=1, n_clusters_per_class=1, n_classes=3, n_informative=2)
    return X, y


# --- Real-world data sets ----------------------------------------------------

def data_iris() -> tuple[np.ndarray, np.ndarray]:
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X.astype(float), y


def data_wine() -> tuple[np.ndarray, np.ndarray]:
    wine = load_wine()
    X = wine.data
    y = wine.target
    return X.astype(float), y


def data_seeds() -> tuple[np.ndarray, np.ndarray]:
    seeds = fetch_openml("seeds", version=1, as_frame=False)
    X = seeds.data
    y = seeds.target
    return X.astype(float), y


def data_ionosphere() -> tuple[np.ndarray, np.ndarray]:
    ionosphere = fetch_openml("ionosphere", version=1, as_frame=False)
    X = ionosphere.data
    y = ionosphere.target
    y = np.where(y == "g", 1, 0)
    return X.astype(float), y


def data_breast_cancer() -> tuple[np.ndarray, np.ndarray]:
    bc = load_breast_cancer()
    X = bc.data
    y = bc.target
    return X.astype(float), y


def data_penguins():
    df = sns.load_dataset("penguins").dropna()
    X = df.select_dtypes(include=[np.number]).to_numpy()
    y = pd.Categorical(df["species"]).codes
    return X, y


def data_mall():
    try:
        df = pd.read_csv("./resources/mall_customers.csv")
        X = df[["Annual Income (k$)", "Spending Score (1-100)"]].to_numpy()
        y = np.zeros(X.shape[0], dtype=int)
        return X.astype(float), y
    except Exception as e:
        ui.notification_show(f"Error loading mall data: {e}", type="error")
        return np.empty((0, 0)), np.empty(0, dtype=int)


def data_spotify():
    try:
        df = pd.read_csv("./resources/spotify_data.csv")
        X = df.to_numpy()
        y = np.zeros(X.shape[0], dtype=int)
        return X.astype(float), y
    except Exception as e:
        ui.notification_show(f"Error loading Spotify data: {e}", type="error")
        return np.empty((0, 0)), np.empty(0, dtype=int)


def data_tweets() -> tuple[np.ndarray, np.ndarray]:
    try:
        df = pd.read_csv("./resources/bluesky_posts_embeddings.csv")
        X = df.to_numpy()
        y = np.zeros(X.shape[0], dtype=int)
        return X.astype(float), y
    except Exception as e:
        ui.notification_show(f"Error loading tweets data: {e}", type="error")
        return np.empty((0, 0)), np.empty(0, dtype=int)
    


# --- GMM plotting helpers ----------------------------------------------------

def _cov_to_matrix(cov, cov_type, d):
    """Return a full (d x d) covariance matrix/matrices from GMM covariances."""
    if cov_type == "full":
        return cov
    if cov_type == "tied":
        return np.broadcast_to(cov, (1, d, d))
    if cov_type == "diag":
        return np.array([np.diag(c) for c in cov])
    if cov_type == "spherical":
        k = cov.shape[0] if np.ndim(cov) else 1
        return np.array([np.eye(d) * float(c) for c in np.atleast_1d(cov)])
    raise ValueError(f"Unknown covariance_type {cov_type!r}")


def _project_gmm_to_pca(gm: GaussianMixture, pca: PCA):
    """
    Project GMM means/covariances from feature space (d) to PCA plot space (2).
    Returns (means_2d, covs_2d) with shapes (k, 2) and (k, 2, 2).
    """
    C = pca.components_[:2, :]
    d = gm.means_.shape[1]
    means_2d = (gm.means_ - pca.mean_) @ C.T  # (k, 2)

    covs_full = _cov_to_matrix(gm.covariances_, gm.covariance_type, d)
    if covs_full.shape[0] == 1:  # tied case
        covs_full = np.repeat(covs_full, gm.means_.shape[0], axis=0)
    # Project each covariance: Σ_2d = C Σ C^T
    covs_2d = np.einsum("ij,kjl,ml->kim", C, covs_full, C)  # (k, 2, 2)
    return means_2d, covs_2d


def _draw_gaussian_ellipse(ax, mean2d, cov2d, edge, face_alpha=0.15, n_std=2.0, lw=1.5, z=3):
    """
    Draw an n-std ellipse given 2x2 covariance (positive-definite).
    """
    # Eigen-decomposition
    vals, vecs = np.linalg.eigh(cov2d)
    # Sort largest first
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    # Angle in degrees of the largest eigenvector
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    # Width/height are 2 * n_std * sqrt(eigenvalues)
    width, height = 2 * n_std * np.sqrt(np.maximum(vals, 1e-12))

    e = Ellipse(xy=mean2d, width=width, height=height, angle=angle,
                edgecolor=edge, facecolor=edge, alpha=face_alpha, lw=lw, zorder=z)
    e.set_fill(True)
    ax.add_patch(e)


def _gmm_params_in_plot_space(gm: GaussianMixture, pca: Optional[PCA]):
    """Return (means2d, covs2d, weights) for plotting in the 2-D plot space P."""
    weights = gm.weights_
    k = gm.n_components
    d = gm.means_.shape[1]
    
    covs_full = _cov_to_matrix(gm.covariances_, gm.covariance_type, d)  # (k,d,d) or (1,d,d)
    
    if pca is None:
        means2d = gm.means_  # (k,2) since d==2 here
        covs2d = covs_full  # (k,2,2) or (1,2,2)
    else:
        C = pca.components_[:2, :]  # (2, d)
        means2d = (gm.means_ - pca.mean_) @ C.T  # (k, 2)
        
        if covs_full.shape[0] == 1:  # tied
            covs_full = np.repeat(covs_full, k, axis=0)
        
        covs2d = np.einsum("ij,kjl,ml->kim", C, covs_full, C)  # (k,2,2)
    
    # Ensure covs2d has one matrix per component
    if covs2d.shape[0] == 1:  # tied without PCA path
        covs2d = np.repeat(covs2d, k, axis=0)
    
    # proper regularization for numerical stability
    eps = 1e-5  # Increased from 1e-6
    I = np.eye(2)
    # Add regularization to each covariance matrix
    for i in range(covs2d.shape[0]):
        covs2d[i] = covs2d[i] + eps * I
    
    return means2d, covs2d, weights


def _neglogpdf_mog(Y: np.ndarray, means: np.ndarray, covs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Compute negative log PDF of mixture of Gaussians.
    Optimized for numerical stability and performance.
    """
    k = means.shape[0]
    n = Y.shape[0]
    logs = np.empty((k, n))
    I = np.eye(2)
    eps_reg = 1e-5  # Increased regularization
    
    for i in range(k):
        Σ = covs[i] + eps_reg * I  # Pre-regularize
        try:
            # Check positive definiteness
            sign, logdet = np.linalg.slogdet(Σ)
            if sign <= 0 or not np.isfinite(logdet):
                # Fallback: stronger regularization
                Σ = Σ + 1e-3 * I
                sign, logdet = np.linalg.slogdet(Σ)
                if sign <= 0:
                    # Component is degenerate, use very low probability
                    logs[i] = -1e10
                    continue
            
            # Use Cholesky for stable inversion
            L = np.linalg.cholesky(Σ)
            logdet = 2.0 * np.sum(np.log(np.diag(L)))
            diff = (Y - means[i]).T                  # shape (2, n)
            sol  = np.linalg.solve(L, diff)          # forward solve
            quad = np.sum(sol**2, axis=0)            # squared Mahalanobis
            logs[i] = -0.5*(2*np.log(2*np.pi) + logdet + quad) + np.log(max(weights[i], 1e-10))

            
        except np.linalg.LinAlgError:
            # If still fails, mark component as degenerate
            logs[i] = -1e10
    
    # Check for all-invalid case
    if not np.any(np.isfinite(logs)):
        return np.full(n, 1e10)
    
    return -logsumexp(logs, axis=0)


def _empty_fig(msg: str, h=500):
    return (go.Figure()
            .add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            .update_layout(height=h, margin=dict(l=20, r=20, t=40, b=20),
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"))


# --- Scatter plot helper -----------------------------------------------------

def _add_scatter(fig: go.Figure, x, y, color, hovertext=None, name="points"):
    # Auto-switch to WebGL for large N
    cls = go.Scattergl if len(x) > 20000 else go.Scatter
    fig.add_trace(cls(x=x, y=y, mode="markers", marker=dict(size=6, color=color),
                      text=hovertext, hoverinfo="text", name=name, showlegend=False))



# --- App ---------------------------------------------------------------------

# ---- UI components ----------------------------------------------------------

def help_card(title: str, md: str):
    return ui.card(ui.card_header(title), ui.markdown(md))

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
    /* Optional: remove background highlight so it blends fully */
    .tab-content {
        background-color: transparent !important;
    }
    """),
    ui.tags.style("""
    /* Overlay container for the plot */
    .plot-overlay-container { position: relative; }

    /* The small floating control box */
    .plot-overlay-controls {
    position: absolute; 
    top: 8px; 
    left: 8px; 
    z-index: 10;
    background: rgba(255,255,255,0.85);
    border-radius: 8px;
    padding: 8px 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    }

    /* Tighter spacing for the two checkboxes */
    .plot-overlay-controls .form-check { margin-bottom: 4px; }
    """),

    ui.tags.style("""
    /* Let flex centering work by disabling absolute fill on plots */
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

app_ui = ui.page_fluid(
    head,
    ui.h2("Hands-on AI I - Unit 3: Unsupervised Learning", class_="text-center mb-3"),
    ui.navset_tab(
        ui.nav_panel("Introduction",
            ui.tags.div(style="margin-top: 15px;"),
            ui.h3("Basics of Unsupervised Learning and Clustering"),
            ui.p("A data set \(X\) typically consists of a set of features, where a concrete observation (sample) \(\mathbf{x}_i\) is the \(i\) th entry in the data set. One sample is characterized by a feature vector: \(\\mathbf{x}_i = (x_{i1}, \\ldots, x_{ij})\), where \(j\) is the number of dimensions."),
            ui.p("In supervised learning we have the convenience of another (special) feature: a target \(Y = (Y_1, \\ldots, Y_K)\) where \(K\) is the number of groups. For example, for the iris data set \(K=3\) (the three kinds of flowers: setosa, virginica and versicolor). When we now train a model using these feature/target combinations \((\\mathbf{x}_1, y_1) \\ldots (\\mathbf{x}_N, y_N)\) given our data set size \(N\), the target feature basically acts like a feedback signal - we can check the prediction \(\hat{Y}\) of the model, because we know what the actual value for \(Y\) is supposed to be. That is why we call this process supervised learning."),
            ui.p("In unsupervised learning we unfortunately do not have this convenience - we only have the plain set of features of our data set \(X\) without the target. This raises quite a few questions:"),
            ui.tags.ul(
                ui.tags.li("Is there a group structure in the data set to begin with?"),
                ui.tags.ul(
                    ui.tags.li("If yes, how many groups are there?"),
                    ui.tags.li("If yes, how are these groups structured (do they have equal shape and form, are they clearly separated, etc.)"),
                ),
                ui.tags.li("Which algorithm should I use?"),
                ui.tags.li("How do I evaluate whether the solution obtained is actually correct?")
            ),
            ui.p("As we will see, the answer to some of these questions is not trivial. While there are several sub-areas of unsupervised learning we will limit our discussion here to clustering, which is a technique that groups similar data points together based on their features (essentially unsupervised classification). It is widely used in various applications such as market segmentation, social network analysis, image segmentation, and anomaly detection."),
            ui.p("This application demonstrates several clustering algorithms on artificial and real-world datasets. You can work with synthetic and real-world datasets, apply different clustering methods, check model validity, optimize hyperparameters, and visualize the results."),
            help_card("Clustering Algorithms",
            """
            - **K-Means**: Partitions data into k clusters by minimizing the variance within each cluster.
            - **Hierarchical Clustering**: Builds a hierarchy of clusters using either agglomerative (bottom-up) or divisive (top-down) approaches (here we only use agglomerative).
            - **DBSCAN**: Density-Based Spatial Clustering of Applications with Noise; groups together points that are closely packed together, marking points in low-density regions as outliers.
            - **Gaussian Mixture Models (GMM)**: Assumes data is generated from a mixture of several Gaussian distributions and uses the Expectation-Maximization algorithm to find the parameters.
            """)
        ),
        ui.nav_panel("Data Selection",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h4("Data Settings"),
                    ui.input_select("data_kind", "Dataset", {
                        "blobs": "Blobs (default)",
                        "gaussian": "Gaussian Blobs 1",
                        "gaussian2": "Gaussian Blobs 2",
                        "moons": "Two Moons",
                        "circles": "Concentric Circles",
                        "uniform": "Uniform Noise",
                        "none": "------------------",
                        "iris": "Iris (real data)",
                        "wine": "Wine (real data)",
                        "breast_cancer": "Breast Cancer (real data)",
                        "ionosphere": "Ionosphere (real data)",
                        "seeds": "Seeds (real data)",
                        "tweets": "Bluesky Posts (real data)",
                        "penguins": "Penguins (real data)",
                        "mall": "Mall Customers (real data)",
                        "spotify": "Spotify Songs (real data)",
                        "none2": "------------------",
                        "csv": "Upload CSV File",
                    }, selected="blobs"),
                    ui.panel_conditional(
                        "input.data_kind == 'blobs'",
                        ui.input_slider("b_n", "N (points)", 50, 1000, 300, step=50),
                        ui.input_numeric("b_centers", "Number of Centers", 4, min=1, step=1),
                        ui.input_slider("b_separation", "Center Range", 0.1, 5.0, 3.5, step=0.1),
                        ui.input_slider("b_std", "Cluster stddev", 0.1, 2.0, 0.60, step=0.05),
                    ),
                    ui.panel_conditional(
                        "input.data_kind == 'moons'",
                        ui.input_slider("m_n", "N (points)", 50, 1000, 300, step=50),
                        ui.input_slider("m_std", "Noise", 0.0, 0.5, 0.10, step=0.05)
                    ),
                    ui.panel_conditional(
                        "input.data_kind == 'circles'",
                        ui.input_slider("c_n", "N (points)", 50, 1000, 300, step=50),
                        ui.input_slider("c_std", "Noise", 0.0, 0.5, 0.10, step=0.05),
                        ui.input_slider("c_factor", "Inner circle size", 0.1, 0.9, 0.5, step=0.05),
                    ),
                    ui.panel_conditional(
                        "input.data_kind == 'gaussian'",
                        ui.input_slider("bg_n", "N (points)", 50, 1000, 300, step=50),
                    ),
                    ui.input_checkbox("scale", "Standardize", True),
                    ui.panel_conditional("input.data_kind == 'csv'",
                        ui.input_file("csv_file", "Upload CSV File (includes header)"),
                        ui.input_checkbox("csv_header", "File has header row", True),
                        ui.input_checkbox("includes_label", "Includes Label (last column)", False),
                    ),
                    ui.hr(),
                    ui.input_numeric("data_seed", "Random Seed", 10, min=1, step=1),
                    ui.hr(),
                    ui.input_action_button("generate_data", "Load Data", class_="btn-primary"),
                ),
                ui.navset_card_pill(
                    ui.nav_panel("Data Plot (True Groups)",
                        ui.panel_conditional("input.generate_data > 0",
                            ui.panel_conditional("input.data_kind != 'tweets'",
                                ui.input_checkbox("color_by_group", "Color by true group", False),
                            ),
                            output_widget("data_plot", height="700px")
                        )
                    ),
                    ui.nav_panel("Table View",
                        ui.panel_conditional("input.generate_data > 0",
                            ui.output_data_frame("data_table"),
                        )
                    ),
                    ui.nav_panel("Data Summary",
                        ui.panel_conditional("input.generate_data > 0",
                            ui.output_text_verbatim("data_summary"),
                        )
                    )
                )
            )
        ),
        ui.nav_panel("Cluster Analysis",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h4("Settings"),
                    ui.input_select("clustering_method", "Algorithm", {
                        "km": "K-Means",
                        "hc": "Hierarchical (Agg)",
                        "db": "DBSCAN",
                        "gm": "Gaussian Mixture (GMM)",
                    }, selected="km"),
                    ui.panel_conditional(
                        "input.clustering_method == 'db'",
                        ui.input_numeric("eps", "Epsilon (eps)", 0.1, min=0.01, step=0.01),
                        ui.input_numeric("min_samples", "Min Samples", 1, min=1, step=1),
                    ),
                    ui.panel_conditional(
                        "input.clustering_method == 'hc'",
                        ui.input_select("linkage", "Linkage", {
                            "ward": "Ward",
                            "complete": "Complete",
                            "average": "Average",
                            "single": "Single",
                        }, selected="ward"),
                        ui.input_numeric("hn_clusters", "Number of Clusters", 1, min=1, step=1),
                    ),
                    ui.panel_conditional(
                        "input.clustering_method == 'km'",
                        ui.input_numeric("n_clusters", "Number of Clusters (k)", 2, min=1, step=1),
                        ui.input_select("init", "Initialization Method", {
                            "k-means++": "k-means++",
                            "random": "Random"
                        }, selected="k-means++"),
                    ),
                    ui.panel_conditional(
                        "input.clustering_method == 'gm'",
                        ui.input_numeric("n_components", "Number of Components", 2, min=1, step=1),
                        ui.input_select("covariance_type", "Covariance Type", {
                            "full": "Full",
                            "tied": "Tied",
                            "diag": "Diagonal",
                            "spherical": "Spherical"
                        }, selected="full"),
                    ),
                    ui.hr(),
                    ui.input_numeric("seed", "Random Seed", 10, min=1, step=1),
                    ui.hr(),
                    ui.input_action_button("run_clustering", "Run Clustering", class_="btn-primary"),
                ),
                ui.navset_card_pill(
                    ui.nav_panel("Cluster Results Plot",
                        ui.panel_conditional(
                            "input.clustering_method != 'gm'",
                            ui.panel_conditional("input.clustering_method == 'km'",
                                ui.row(
                                    ui.input_checkbox("show_voronoi", "Show Voronoi", False),
                                    ui.input_checkbox("show_centroids", "Show Centroids", False),
                                )
                            ),
                            ui.panel_conditional("input.clustering_method == 'hc'",
                                ui.row(
                                    ui.output_ui("show_decision_regions_ag")
                                )
                            ),
                            
                            ui.panel_conditional("input.clustering_method == 'db'",
                                ui.row(
                                    ui.input_checkbox("show_epsilon", "Show Epsilon", False),
                                    ui.output_ui("show_decision_regions_db")
                                ),
                            ),
                            output_widget("clustering_plot", height="700px"),
                        ),
                        ui.panel_conditional(
                            "input.clustering_method == 'gm' && input.generate_data > 0",
                            ui.row(
                                ui.input_checkbox("show_contour", "Show Contour", False),
                                ui.input_checkbox("gm_show_ellipses", "Show component ellipses (2σ)", False),
                            ),
                            output_widget("clustering_plot_gmm", height="700px"),
                        ), 
                    ),
                    ui.nav_panel("Cluster Result Details",
                        ui.accordion(
                            ui.accordion_panel("Table with Cluster Labels",
                                ui.output_data_frame("cluster_labels"),
                            ), open=False
                        ),
                        ui.accordion(
                            ui.accordion_panel("Cluster Sizes Summary",
                                ui.output_data_frame("cluster_sizes"),
                            ), open=False
                        ),
                    ),
                    ui.nav_panel("Cluster Validation",
                        ui.output_text_verbatim("clustering_info"),
                        ui.panel_conditional("input.clustering_method != 'db' && input.clustering_method != 'gm'",
                            ui.tags.div(style="margin-top: 15px;"),
                            output_widget("silhouette", height="600px")
                        )
                    ),
                    ui.nav_panel("Cluster Hyperparameter Tuning",
                        ui.panel_conditional(
                            "input.clustering_method == 'hc'",
                            
                            output_widget("linkage_plot", height="700px"),
                            ui.accordion(
                                ui.accordion_panel("Linkage Matrix Info",
                                    ui.output_data_frame("linkage_info"),
                                    expanded=False
                                )
                            ),
                        ),
                        ui.panel_conditional("input.clustering_method == 'km'",
                            ui.input_slider("elbow_kmax", "Max k for Elbow Method", 2, 20, 10, step=1),
                            output_widget("elbow_plot", height="500px")
                        ),
                        ui.panel_conditional("input.clustering_method == 'gm'",
                            ui.input_slider("aic_bic_kmax", "Max k for AIC/BIC", 2, 20, 10, step=1),
                            output_widget("aic_bic_plot", height="500px")
                        ),
                        ui.panel_conditional("input.clustering_method == 'db'",
                            ui.input_slider("dbscan_k", "k for k-NN distance plot", 2, 20, 5, step=1),
                            output_widget("dbscan_knn_plot", height="500px"),
                        )
                    ),
                    ui.nav_panel("Sample from Clusters",
                        ui.panel_conditional("input.data_kind == 'csv'",
                            ui.input_file("auxiliary_csv", "Upload Auxiliary CSV File (if available, nrow must match)", accept=[".csv"]),
                        ),
                        ui.row(
                            ui.input_numeric("sample_cluster_nr", "Cluster Label to Investigate", 0, min=0, step=1),
                            ui.input_numeric("sample_per_cluster", "Number of Samples per Cluster", 1, min=1, step=1),
                        ),
                        ui.output_data_frame("cluster_samples")
                    )
                )
            )
        ),
        ui.nav_panel("Application: Image Segmentation",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_file("image_file", "Upload Image", accept=[".png", ".jpg", ".jpeg"]),
                    ui.input_slider("n_segments", "Number of Segments (k)", 2, 20, 5, step=1),
                    ui.hr(),
                    ui.input_action_button("run_segmentation", "Run Segmentation", class_="btn-primary"),
                ),
                ui.card(
                    ui.card_header("Image Segmentation Results"),
                    ui.output_plot("segmentation_plot", height="500px")
                )
            )
        )
    )
)


# ---- Server -----------------------------------------------------------------

def server(input, output, session):

    km_centroids = reactive.Value(None)
    km_model = reactive.Value(None)
    db_model = reactive.Value(None)
    gmm_model = reactive.Value(None)
    plot_pca = reactive.Value(None)           # fitted PCA for the current data (or None)
    km_centroids_plot = reactive.Value(None)  # KMeans centroids projected to plot space (or None)
    gmm_model_plot = reactive.Value(None)     # 2-D GMM fitted on plot space (for contours when d>2)
    linkage_data = reactive.Value(None)
    feature_names = reactive.Value(None)
    target_names = reactive.Value(None)


    plot_bump = reactive.Value(0)

    def _bump_plot():
        try:
            plot_bump.set(plot_bump() + 1)
        except Exception:
            pass

    def _watch(ev):
        @reactive.Effect
        @reactive.event(ev, ignore_init=True)
        def _inner():
            # bump plot re-render on the respective event
            _bump_plot()

    # watchers for each optional overlay control
    _watch(input.show_regions_ag)
    _watch(input.show_regions_db)
    _watch(input.show_voronoi)
    _watch(input.show_centroids)
    _watch(input.show_epsilon)

    def _prepare_spaces() -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[PCA]]:
        """
        Returns (Xc, y, P, pca):
        Xc  = feature space used for clustering
        y   = true labels if available
        P   = 2-D plot space (==X if already 2D; PCA(Xc) if >2D)
        pca = PCA object used to obtain P (None if no PCA needed)
        """
        X, y = data()
        if X.ndim != 2 or X.shape[1] == 0:
            return np.empty((0, 0)), np.empty(0), np.empty((0, 0)), None

        Xc = StandardScaler().fit_transform(X) if input.scale() else X

        if Xc.shape[1] > 2:
            pca = PCA(n_components=2, random_state=input.seed())
            P = pca.fit_transform(Xc)
            plot_pca.set(pca)
        else:
            pca = None
            P = Xc
            plot_pca.set(None)

        return Xc, y, P, pca

        
    @reactive.effect
    def _reset_models_on_data_change():
        _ = (
            input.data_kind(), input.scale(), input.data_seed(),
            input.b_n(), input.b_centers(), input.b_separation(), input.b_std(),
            input.m_n(), input.m_std(),
            input.c_n(), input.c_std(), input.c_factor(),
            input.bg_n(),
            input.csv_file()
        )
        km_centroids.set(None)
        km_centroids_plot.set(None)
        km_model.set(None)
        db_model.set(None)
        gmm_model.set(None)
        linkage_data.set(None)
        region_cache.set({"bbox": None, "n": None, "xs": None, "ys": None, "XX": None, "YY": None, "grid": None})

    @reactive.effect
    def _disable_decision_regions_if_pca():
        if plot_pca() is not None:
            session.send_input_message("show_regions_ag", {"value": False})
            session.send_input_message("show_regions_db", {"value": False})
            session.send_input_message("show_regions_ag", {"disabled": True})
            session.send_input_message("show_regions_db", {"disabled": True})
        else:
            session.send_input_message("show_regions_ag", {"disabled": False})
            session.send_input_message("show_regions_db", {"disabled": False})

    @reactive.effect
    @reactive.event(input.clustering_method)
    def _reset_decision_checkboxes():
        method = input.clustering_method()
        # Always start with a clean slate
        session.send_input_message("show_regions_ag", {"value": False})
        session.send_input_message("show_regions_db", {"value": False})
        session.send_input_message("show_voronoi", {"value": False})
        session.send_input_message("show_centroids", {"value": False})
        session.send_input_message("show_epsilon", {"value": False})

        # Optionally: disable irrelevant checkboxes so they don't remain greyed out
        if method == "hc":
            session.send_input_message("show_regions_ag", {"disabled": False})
        elif method == "db":
            session.send_input_message("show_regions_db", {"disabled": False})
        elif method == "km":
            session.send_input_message("show_voronoi", {"disabled": False})
            session.send_input_message("show_centroids", {"disabled": False})
        elif method == "gm":
            session.send_input_message("show_contour", {"disabled": False})
            session.send_input_message("gm_show_ellipses", {"disabled": False})


    @output
    @render.ui
    def show_decision_regions_ag():
        if plot_pca() is not None:
            return None
        return ui.input_checkbox("show_regions_ag", "Show decision regions (2D only)", False)

    @output
    @render.ui
    def show_decision_regions_db():
        if plot_pca() is not None:
            return None
        return ui.input_checkbox("show_regions_db", "Show decision regions (2D only)", False)

    @reactive.calc
    def data() -> tuple[np.ndarray, np.ndarray]:
        _ = input.generate_data()

        if input.data_kind() == "blobs":
            X, y = data_blobs(n=input.b_n(), centers=input.b_centers(), std=input.b_std(), seed=input.data_seed(), separation=input.b_separation())
            feature_names.set(None)
            target_names.set(None)
        elif input.data_kind() == "moons":
            X, y = data_moons(n=input.m_n(), noise=input.m_std(), seed=input.data_seed())
            feature_names.set(None)
            target_names.set(None)
        elif input.data_kind() == "circles":
            X, y = data_circles(n=input.c_n(), noise=input.c_std(), factor=input.c_factor(), seed=input.data_seed())
            feature_names.set(None)
            target_names.set(None)
        elif input.data_kind() == "gaussian":
            X, y = gaussian_blobs(n=input.bg_n(), seed=input.data_seed())
            feature_names.set(None)
            target_names.set(None)
        elif input.data_kind() == "uniform":
            X, y = data_uniform(n=300, seed=input.data_seed())
            feature_names.set(None)
            target_names.set(None)
        elif input.data_kind() == "iris":
            iris = load_iris()
            X = iris.data.astype(float)
            y = iris.target
            target_names.set(list(iris.target_names))
            feature_names.set(list(iris.feature_names))
        elif input.data_kind() == "wine":
            wine = load_wine()
            X = wine.data.astype(float)
            y = wine.target
            target_names.set(list(wine.target_names))
            feature_names.set(list(wine.feature_names))
        elif input.data_kind() == "tweets":
            X, y = data_tweets()
            target_names.set(None)
            feature_names.set(None)
        elif input.data_kind() == "seeds":
            seeds = fetch_openml("seeds", version=1, as_frame=True)
            df = seeds.frame.drop(columns=["Class"], errors="ignore")
            X, y = df.to_numpy(dtype=float), pd.Categorical(seeds.frame["Class"]).codes
            feature_names.set(list(df.columns))
            target_names.set(list(seeds.frame["Class"].cat.categories))
        elif input.data_kind() == "ionosphere":
            iono = fetch_openml("ionosphere", version=1, as_frame=True)
            df = iono.frame.drop(columns=["class"], errors="ignore")
            X = df.to_numpy(dtype=float)
            y = np.where(iono.frame["class"] == "g", 1, 0)
            feature_names.set(list(df.columns))
            target_names.set(["b", "g"])
        elif input.data_kind() == "breast_cancer":
            bc = load_breast_cancer()
            X, y = bc.data.astype(float), bc.target
            feature_names.set(list(bc.feature_names))
            target_names.set(list(bc.target_names))
        elif input.data_kind() == "penguins":
            df = sns.load_dataset("penguins").dropna()
            num = df.select_dtypes(include=[np.number])
            X, y = num.to_numpy(), pd.Categorical(df["species"]).codes
            feature_names.set(list(num.columns))
            target_names.set(list(df["species"].unique()))
        elif input.data_kind() == "mall":
            df = pd.read_csv("./resources/mall_customers.csv")
            cols_keep = ["Annual Income (k$)", "Spending Score (1-100)"]
            X = df[cols_keep].to_numpy(dtype=float)
            y = np.zeros(X.shape[0], dtype=int)
            feature_names.set(cols_keep)
            target_names.set(None)
        elif input.data_kind() == "spotify":
            X, y = data_spotify()
            feature_names.set(None)
            target_names.set(None)
        elif input.data_kind() == "none" or input.data_kind() == "none2":
            X, y = np.empty((0, 0)), np.empty(0, dtype=int)
            feature_names.set(None)
            target_names.set(None)
            return X, y
        elif input.data_kind() == "gaussian2":
            X, y = gaussian_blobs2(seed=input.data_seed())
            feature_names.set(None)
            target_names.set(None)
        elif input.data_kind() == "csv":
            file_info = input.csv_file()
            if file_info is None:
                ui.notification_show("Please upload a CSV file.", type="warning")
                return np.empty((0, 0)), np.empty(0, dtype=int)
            try:
                file_info = input.csv_file()
                if file_info is None:
                    ui.notification_show("Please upload a CSV file.", type="warning")
                    return np.empty((0, 0)), np.empty(0, dtype=int)

                fi = file_info[0]
                path = fi["datapath"]

                if not input.csv_header():
                    df = pd.read_csv(path, header=None, sep=None, engine="python")
                    n_cols = df.shape[1]
                    if input.includes_label():
                        feature_names.set(None)
                else:
                    df = pd.read_csv(path, sep=None, engine="python")
                    df.columns = df.columns.str.strip()
                    feature_names.set(list(df.columns[:-1]) if input.includes_label() else list(df.columns))
                    

                # drop non-numeric columns (ids, text)
                num = df.select_dtypes(include=[np.number])
                if num.shape[1] >= 2:
                    X = num.to_numpy()
                else:
                    # coerce everything to numeric (errors to NaN), drop NaN cols
                    coerced = df.apply(pd.to_numeric, errors="coerce")
                    X = coerced.dropna(axis=1, how="any").to_numpy()

                if X.shape[0] == 0 or X.shape[1] == 0:
                    ui.notification_show(
                        "Uploaded CSV file contains no usable numeric columns (need at least 2).",
                        type="error"
                    )
                    return np.empty((0, 0)), np.empty(0, dtype=int)
                if input.includes_label():
                    y = X[:, -1].astype(int)
                    X = X[:, :-1]
                    target_names.set([str(l) for l in np.unique(y)])
                else:
                    y = np.zeros(X.shape[0], dtype=int)
                    target_names.set(None)
                return X.astype(float), y
            except Exception as e:
                ui.notification_show(f"Error reading CSV file: {e}", type="error")
                return np.empty((0, 0)), np.empty(0, dtype=int)
        else:
            raise ValueError("Unknown data kind")
        return X, y
    

    @reactive.event(input.run_clustering)
    def clustering() -> np.ndarray:
        Xc, _, P, pca = _prepare_spaces()
        if Xc is None or Xc.size == 0 or Xc.shape[0] < 2:
            ui.notification_show("Not enough data to fit a model.", type="error")
            return np.array([], dtype=int)
        method = input.clustering_method()
        seed = input.seed()

        if method == "km":
            model = KMeans(n_clusters=input.n_clusters(), init=input.init(), random_state=seed)
        elif method == "hc":
            model = AgglomerativeClustering(n_clusters=input.hn_clusters(), linkage=input.linkage())
        elif method == "db":
            model = DBSCAN(eps=input.eps(), min_samples=input.min_samples())
        elif method == "gm":
            model = GaussianMixture(n_components=input.n_components(), covariance_type=input.covariance_type(), random_state=seed)
        else:
            raise ValueError("Unknown clustering method")

        model.fit(Xc)

        if method == "km":
            km_model.set(model)
            km_centroids.set(model.cluster_centers_)
            if pca is not None:
                km_centroids_plot.set(pca.transform(model.cluster_centers_))
            else:
                km_centroids_plot.set(model.cluster_centers_)
            labels = model.labels_
        elif method == "gm":
            gmm_model.set(model)
            labels = model.predict(Xc)
        elif method == "db":
            db_model.set(model)
            labels = model.labels_
        else:
            labels = model.labels_

        return labels

    # helper function for the plotly scatter point colors
    @reactive.calc
    def point_colors() -> list[str]:
        Xc, y, P, _ = _prepare_spaces()
        labels = y if input.color_by_group() else np.zeros(Xc.shape[0], dtype=int)
        k = int(len(np.unique(labels)))
        if k <= 10:
            palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
                    "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"][:k]
        elif k <= 20:
            cs = [plt.cm.tab20(i) for i in range(min(20, k))]
            palette = [f"rgba({int(255*r)},{int(255*g)},{int(255*b)},{a})" for (r,g,b,a) in cs]
        else:
            cs = [plt.cm.hsv(i/k) for i in range(k)]
            palette = [f"rgba({int(255*r)},{int(255*g)},{int(255*b)},{a})" for (r,g,b,a) in cs]
            
        point_colors = [palette[int(l) % len(palette)] for l in labels]
        return point_colors


    @output
    @render_plotly
    def data_plot():
        # feature space (Xc), true labels (y), 2D plot space (P), and PCA object (pca)
        Xc, y, P, _ = _prepare_spaces()

        if P is None or P.size == 0 or P.shape[1] < 2:
            return _empty_fig("No data (or fewer than 2 numeric columns) to plot.")

        hover_texts = [
            (f"Index: {i}<br>x1: {P[i,0]:.3f}<br>x2: {P[i,1]:.3f}" +
            (f"<br>True label: {y[i]}" if input.color_by_group() else ""))
            for i in range(P.shape[0])
        ]

        color = "blue" if not input.color_by_group() else point_colors()

        fig = go.Figure()
        _add_scatter(fig, P[:, 0], P[:, 1], color, hover_texts)

        if plot_pca() is not None:
            fig.add_annotation(
                text="PCA → 2D",
                xref="paper", yref="paper", x=0.99, y=0.99,
                showarrow=False, xanchor="right", yanchor="top",
                font=dict(size=11), bgcolor="rgba(255,255,255,0.7)"
            )

        fig.update_layout(
            height=700,
            xaxis_title="x1", yaxis_title="x2",
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            hovermode="closest",
        )
        fig.update_xaxes(showline=True, linewidth=2, linecolor="black", mirror=True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor="black", mirror=True)

        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1
        )
        fig.update_xaxes(constrain="domain")

        return fig


    @output
    @render_plotly
    @reactive.event(input.run_clustering, plot_bump)  # bump on overlay changes
    def clustering_plot():

        Xc, _, P, pca = _prepare_spaces()
        labels = clustering()
        method = input.clustering_method()

        try:
            show_ag = bool(input.show_regions_ag())
        except Exception:
            show_ag = False
        try:
            show_db = bool(input.show_regions_db())
        except Exception:
            show_db = False

        if not input.run_clustering() or input.run_clustering() == 0:
            return _empty_fig("Click “Run Clustering” to see the results.", h=700)

        if P is None or P.size == 0 or P.shape[1] < 2:
            return _empty_fig("No data (or fewer than 2 numeric columns) to plot.", h=700)

        if method == "gm":
            return go.Figure().update_layout(height=700, margin=dict(l=20, r=20, t=20, b=20))

        def _palette(k: int) -> list[str]:
            if k <= 10:
                return ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
                        "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"][:k]
            elif k <= 20:
                cs = [plt.cm.tab20(i) for i in range(min(20, k))]
                return [f"rgba({int(255*r)},{int(255*g)},{int(255*b)},{a})" for (r,g,b,a) in cs]
            else:
                cs = [plt.cm.hsv(i/k) for i in range(k)]
                return [f"rgba({int(255*r)},{int(255*g)},{int(255*b)},{a})" for (r,g,b,a) in cs]

        def _lighten(hex_or_rgba: str, amount: float = 0.85) -> str:
            if hex_or_rgba.startswith("#"):
                r = int(hex_or_rgba[1:3], 16)/255
                g = int(hex_or_rgba[3:5], 16)/255
                b = int(hex_or_rgba[5:7], 16)/255
                a = 1.0
            else:
                vals = hex_or_rgba.strip()[5:-1].split(",")
                r, g, b, a = [float(vals[0])/255, float(vals[1])/255, float(vals[2])/255, float(vals[3])]
            r = r + (1-r)*amount
            g = g + (1-g)*amount
            b = b + (1-b)*amount
            return f"rgba({int(255*r)},{int(255*g)},{int(255*b)},{a})"

        def _discrete_colorscale(n: int, base: list[str]) -> list[tuple[float, str]]:
            cs = []
            for i in range(n):
                t0 = i / n
                t1 = (i + 1) / n
                cs.append((t0, base[i % len(base)]))
                cs.append((t1, base[i % len(base)]))
            return cs

        def _add_region_heatmap(fig: go.Figure, Z: np.ndarray, palette: list[str]):
            cs = _discrete_colorscale(int(np.max(Z) + 1), [_lighten(c, 0.85) for c in palette])
            fig.add_trace(go.Heatmap(
                x=xs, y=ys, z=Z, showscale=False, colorscale=cs, opacity=0.55,
                zauto=False, zmin=0, zmax=int(np.max(Z)), hoverinfo="skip"
            ))
            if len(xs) > 1 and len(ys) > 1:
                dx = xs[1] - xs[0]
                dy = ys[1] - ys[0]
                fig.update_xaxes(range=[xs[0] - dx/2, xs[-1] + dx/2])
                fig.update_yaxes(range=[ys[0] - dy/2, ys[-1] + dy/2])
            fig.data[-1].zsmooth = False 

        def _add_scatter(fig: go.Figure, x, y, colors, names=None, name="points"):
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="markers",
                marker=dict(size=7, line=dict(width=0.5, color="rgba(0,0,0,0.25)"),
                            color=colors),
                name=name, hoverinfo="skip" if names is None else "text",
                text=names
            ))

        def _knn_regions(labels_arr: np.ndarray, ignore_label=None, k: int = 1):
            from sklearn.neighbors import KNeighborsClassifier
            L = np.asarray(labels_arr)
            mask = np.ones(L.shape[0], dtype=bool)
            if ignore_label is not None:
                mask &= (L != ignore_label)
            if not np.any(mask):
                return None
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(P[mask], L[mask])
            Z = knn.predict(grid).reshape(XX.shape)
            return Z

        xs = ys = XX = YY = grid = None
        def _ensure_grid():
            nonlocal xs, ys, XX, YY, grid
            if grid is None:
                xs, ys, XX, YY, grid = _get_region_grid(P, n=300, pad_ratio=0.05)

        fig = go.Figure().update_layout(height=700, margin=dict(l=20, r=20, t=20, b=20))

        k = int(len(np.unique(labels[labels != -1]))) if method == "db" else int(len(np.unique(labels)))
        palette = _palette(max(k, 1))
        point_colors = [palette[int(l) % len(palette)] for l in labels]
        if method == "db":
            for i, l in enumerate(labels):
                if l == -1:
                    point_colors[i] = "rgba(0,0,0,0.65)"  # noise dark grey

        if method == "km" and km_centroids() is not None:
            C_plot = km_centroids_plot()

            if input.show_voronoi() and C_plot is not None and C_plot.shape[1] == 2 and C_plot.shape[0] >= 3:
                try:
                    _ensure_grid()
                    d2 = ((grid[:, None, :] - C_plot[None, :, :])**2).sum(axis=2)
                    region = d2.argmin(axis=1).reshape(XX.shape)
                    _add_region_heatmap(fig, region, palette)

                    vor = Voronoi(C_plot)
                    for (i_r, j_r) in vor.ridge_vertices:
                        if i_r == -1 or j_r == -1:
                            continue
                        p0, p1 = vor.vertices[i_r], vor.vertices[j_r]
                        fig.add_trace(go.Scatter(
                            x=[p0[0], p1[0]], y=[p0[1], p1[1]],
                            mode="lines", line=dict(width=2, color="orange"),
                            hoverinfo="skip", showlegend=False
                        ))
                except Exception:
                    fig.add_annotation(
                        text="Voronoi failed (degenerate centroid layout).",
                        xref="paper", yref="paper", x=0.01, y=0.99, showarrow=False,
                        xanchor="left", yanchor="top", font=dict(size=11)
                    )
            elif input.show_voronoi() and input.n_clusters() < 3:
                fig.add_annotation(
                    text="Voronoi requires at least 3 centroids.",
                    xref="paper", yref="paper", x=0.01, y=0.99,
                    showarrow=False, xanchor="left", yanchor="top", font=dict(size=11)
                )

            idx_points = len(fig.data)
            _add_scatter(fig, P[:, 0], P[:, 1], point_colors, None, name="points")

            if input.show_centroids() and C_plot is not None:
                fig.add_trace(go.Scatter(
                    x=C_plot[:, 0], y=C_plot[:, 1], mode="markers",
                    marker=dict(symbol="x", size=14, line=dict(width=2), color="white"),
                    name="centroids", hoverinfo="skip"
                ))

            hover_texts = [
                f"Index: {i}<br>x1: {P[i,0]:.3f}<br>x2: {P[i,1]:.3f}<br>Assigned cluster: {int(labels[i])}"
                for i in range(P.shape[0])
            ]
            fig.data[idx_points].text = hover_texts
            fig.data[idx_points].hoverinfo = "text"

        elif method == "db":# and show_db:
            if show_db and db_model() is not None:
                _ensure_grid()
                db = db_model()
                eps = float(getattr(db, "eps", input.eps()))

                labels_arr = np.asarray(labels)
                core_mask = np.zeros(P.shape[0], dtype=bool)  # core samples
                if hasattr(db, "core_sample_indices_") and db.core_sample_indices_ is not None:
                    core_mask[db.core_sample_indices_] = True

                if not np.any(core_mask):
                    Z = np.full(XX.shape, -1, dtype=int)
                else:
                    from sklearn.neighbors import NearestNeighbors
                    cores_P = P[core_mask]
                    cores_lab = labels_arr[core_mask]

                    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
                    nn.fit(cores_P)
                    dist, idx = nn.kneighbors(grid, n_neighbors=1, return_distance=True)
                    idx = idx.ravel()
                    dist = dist.ravel()

                    Z_flat = np.where(dist <= eps, cores_lab[idx], -1)
                    Z = Z_flat.reshape(XX.shape)

                nz = labels_arr[labels_arr != -1]
                k_eff = int(len(np.unique(nz))) or 1
                pal_eff = _palette(k_eff)
                noise_color = "rgba(0,0,0,0.12)"

                # discrete colorscale for z ∈ {-1, 0..k_eff-1}
                # map -1 on first interval [0, 1/(k_eff+1))
                colors = [noise_color] + [_lighten(c, 0.85) for c in pal_eff]
                n_bins = k_eff + 1
                cs = []
                for i in range(n_bins):
                    t0 = i / n_bins
                    t1 = (i + 1) / n_bins
                    cs.append((t0, colors[i]))
                    cs.append((t1, colors[i]))

                fig.add_trace(go.Heatmap(
                    x=xs, y=ys, z=Z,
                    zauto=False, zmin=-1, zmax=max(0, k_eff-1),
                    colorscale=cs, showscale=False, opacity=0.55, hoverinfo="skip"
                ))

            idx_points = len(fig.data)
            _add_scatter(fig, P[:, 0], P[:, 1], point_colors, None, name="points")

            x_rng = y_rng = None

            if input.show_epsilon() and db_model() is not None and pca is None:
                db = db_model()
                eps = float(getattr(db, "eps", input.eps()))

                # store ranges to fix axes later
                x_rng = [float(P[:, 0].min()) - eps, float(P[:, 0].max()) + eps]
                y_rng = [float(P[:, 1].min()) - eps, float(P[:, 1].max()) + eps]

                # draw circles
                core_mask = np.zeros(P.shape[0], dtype=bool)
                if hasattr(db, "core_sample_indices_") and db.core_sample_indices_ is not None:
                    core_mask[db.core_sample_indices_] = True

                if np.any(core_mask):
                    t = np.linspace(0, 2*np.pi, 181)
                    unit = np.column_stack([np.cos(t), np.sin(t)])
                    for (x0, y0) in P[core_mask]:
                        cx, cy = x0 + eps * unit[:, 0], y0 + eps * unit[:, 1]
                        fig.add_trace(go.Scatter(
                            x=cx, y=cy, mode="lines",
                            line=dict(width=2, color="black", dash="dot"),
                            hoverinfo="skip", showlegend=False,
                            cliponaxis=False
                        ))

            elif input.show_epsilon() and pca is not None:
                fig.add_annotation(
                    text="Epsilon circles shown only for true 2-D data.",
                    xref="paper", yref="paper", x=0.01, y=0.99,
                    showarrow=False, xanchor="left", yanchor="top", font=dict(size=11)
                )

            hover_texts = [
                f"Index: {i}<br>x1: {P[i,0]:.3f}<br>x2: {P[i,1]:.3f}<br>Assigned cluster: {int(labels[i])}"
                for i in range(P.shape[0])
            ]
            fig.data[idx_points].text = hover_texts
            fig.data[idx_points].hoverinfo = "text"

        else: # agglomerative
            if show_ag:
                _ensure_grid()
                Z = _knn_regions(labels, ignore_label=None, k=1)
                if Z is not None:
                    _add_region_heatmap(fig, Z, palette)

            idx_points = len(fig.data)
            _add_scatter(fig, P[:, 0], P[:, 1], point_colors, None, name="points")

            hover_texts = [
                f"Index: {i}<br>x1: {P[i,0]:.3f}<br>x2: {P[i,1]:.3f}<br>Assigned cluster: {int(labels[i])}"
                for i in range(P.shape[0])
            ]
            fig.data[idx_points].text = hover_texts
            fig.data[idx_points].hoverinfo = "text"

        keep_aspect = (plot_pca() is None) and (
            (input.clustering_method() == "km" and input.show_voronoi())
            or (input.clustering_method() == "hc" and bool(input.show_regions_ag()))
            or (input.clustering_method() == "db" and bool(input.show_regions_db()))
        )

        if keep_aspect: #  keep 1:1 aspect ratio for Voronoi or regions
            fig.update_xaxes(scaleanchor="y", scaleratio=1)
        else: #  plot freely resized
            fig.update_layout(
                xaxis=dict(constrain=None),
                yaxis=dict(constrain=None)
            )

        fig.update_layout(legend=dict(orientation="h", 
                                      yanchor="bottom", 
                                      y=1.02,
                                      xanchor="right", 
                                      x=1.0))
        fig.update_layout(
            height=700,
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor="rgba(0,0,0,0)", 
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            hovermode="closest",
        )
        fig.update_xaxes(showline=True, linewidth=2, linecolor="black", mirror=True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor="black", mirror=True)

        try:
            if input.clustering_method() == "db" and input.show_epsilon() and pca is None and 'x_rng' in locals() and x_rng and y_rng:
                fig.update_xaxes(range=x_rng, autorange=False)
                fig.update_yaxes(range=y_rng, autorange=False)
        except Exception:
            pass

        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1
        )
        fig.update_xaxes(constrain="domain")
        return fig
    

    @output
    @render_plotly
    @reactive.event(input.run_clustering, input.show_contour, input.gm_show_ellipses)
    def clustering_plot_gmm():
        if input.clustering_method() != "gm":
            return go.Figure().update_layout(height=700, margin=dict(l=20, r=20, t=20, b=20))

        labels = clustering()
        Xc, _, P, pca = _prepare_spaces()
        gm = gmm_model()
        if gm is None or P is None or P.size == 0 or P.shape[1] < 2:
            return _empty_fig("Run GMM on a dataset with ≥ 2 numeric features.", h=700)

        unique_labs = np.unique(labels)
        k = int(len(unique_labs))
        if k <= 10:
            palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
                    "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"][:k]
        elif k <= 20:
            cs = [plt.cm.tab20(i) for i in range(min(20, k))]
            palette = [f"rgba({int(255*r)},{int(255*g)},{int(255*b)},{a})" for (r,g,b,a) in cs]
        else:
            cs = [plt.cm.hsv(i/k) for i in range(k)]
            palette = [f"rgba({int(255*r)},{int(255*g)},{int(255*b)},{a})" for (r,g,b,a) in cs]
        point_colors = [palette[int(l) % len(palette)] for l in labels]

        proba = gm.predict_proba(Xc)  # (n, k)
        def _fmt_probs(p):
            if p.shape[0] <= 6:
                return ", ".join([f"p{i}={p[i]:.3f}" for i in range(p.shape[0])])
            top = np.argsort(p)[::-1][:6]
            return ", ".join([f"p{int(i)}={p[i]:.3f}" for i in top]) + ", …"
        hover_texts = [
            f"Index: {i}<br>x1: {P[i,0]:.3f}<br>x2: {P[i,1]:.3f}<br>{_fmt_probs(proba[i])}<br>Assigned: {int(labels[i])}"
            for i in range(P.shape[0])
        ]

        fig = go.Figure()

        if input.show_contour():
            try:
                means2d, covs2d, weights = _gmm_params_in_plot_space(gm, plot_pca())

                try:
                    xs, ys, XX, YY, grid = _get_region_grid(P, n=150, pad_ratio=0.05)
                except NameError:
                    pad_x = 0.05 * (P[:, 0].ptp() or 1.0)
                    pad_y = 0.05 * (P[:, 1].ptp() or 1.0)
                    xs = np.linspace(P[:, 0].min() - pad_x, P[:, 0].max() + pad_x, 120)
                    ys = np.linspace(P[:, 1].min() - pad_y, P[:, 1].max() + pad_y, 120)
                    XX, YY = np.meshgrid(xs, ys)
                    grid = np.column_stack([XX.ravel(), YY.ravel()])

                Z = _neglogpdf_mog(grid, means2d, covs2d, weights).reshape(XX.shape)

                if np.any(np.isfinite(Z)):
                    fig.add_trace(go.Contour(
                        x=xs, y=ys, z=Z,
                        contours=dict(showlines=True),
                        opacity=0.35, showscale=False, name="-logpdf"
                    ))

                    # projected component means
                    fig.add_trace(go.Scatter(
                        x=means2d[:, 0], y=means2d[:, 1],
                        mode="markers",
                        marker=dict(symbol="x", size=12, line=dict(width=2), color="white"),
                        name="means", hoverinfo="skip", showlegend=False
                    ))

                    # Optional ellipses
                    if input.gm_show_ellipses():
                        def _ellipse_poly(mu, Sigma, nsig=2.0, n=181):
                            vals, vecs = np.linalg.eigh(0.5*(Sigma+Sigma.T))
                            vals = np.clip(vals, 1e-12, None)
                            t = np.linspace(0, 2*np.pi, n)
                            circle = np.vstack([np.cos(t), np.sin(t)])
                            A = vecs @ (np.sqrt(vals)[:, None] * circle)
                            return mu[0] + nsig*A[0], mu[1] + nsig*A[1]

                        for i in range(means2d.shape[0]):
                            ex, ey = _ellipse_poly(means2d[i], covs2d[i], nsig=2.0, n=181)
                            fig.add_trace(go.Scatter(
                                x=ex, y=ey, mode="lines",
                                line=dict(width=1.2),
                                name=f"comp {i} (2σ)", hoverinfo="skip", showlegend=False
                            ))
            except Exception as e:
                print(f"[GMM contour warning] {e}")


        idx_points = len(fig.data)
        _add_scatter(fig, P[:, 0], P[:, 1], point_colors, hover_texts, name="points")
        # ensure hover set on the correct trace even if new traces get added later
        fig.data[idx_points].hoverinfo = "text"


        if pca is not None:
            fig.add_annotation(
                text="PCA → 2D",
                xref="paper", yref="paper", x=0.99, y=0.99,
                showarrow=False, xanchor="right", yanchor="top",
                font=dict(size=11), bgcolor="rgba(255,255,255,0.7)"
            )

        fig.update_layout(
            height=700,
            xaxis_title="x1", yaxis_title="x2",
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor="rgba(0,0,0,0)", 
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            hovermode="closest",
        )
        fig.update_xaxes(showline=True, linewidth=2, linecolor="black", mirror=True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor="black", mirror=True)


        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1
        )
        fig.update_xaxes(constrain="domain")
        return fig


    @output
    @render.text
    @reactive.event(input.run_clustering)
    def clustering_info():
        labels = clustering()
        if labels is None:
            return "Run clustering to see results."
        Xc, _, _, _ = _prepare_spaces()

        if input.clustering_method() == "db":
            return textwrap.dedent(f"""\
            Number of clusters: {len(set(labels)) - (1 if -1 in labels else 0)}
            Number of noise points: {(labels == -1).sum()}

            Silhouette Score: N/A (not defined for clusters with noise)
            Inertia: N/A (not defined for DBSCAN)
            -----------
            """)
        elif input.clustering_method() == "gm":
            gm = gmm_model()
            if gm is None:
                return "Run clustering to see results."
            return textwrap.dedent(f"""\
            Number of components: {gm.n_components}
            Convergence: {'Yes' if gm.converged_ else 'No'}
            Number of iterations: {gm.n_iter_}

            AIC: {gm.aic(Xc):.4f} (lower is better)
            BIC: {gm.bic(Xc):.4f} (lower is better)

            Silhouette Score: N/A (not defined for soft clustering)
            Inertia: N/A (not defined for GMM)
            -----------
            """)
        else:
            if len(set(labels)) < 2:
                return "At least two clusters are required to compute clustering metrics."
            sil_score = silhouette_score(Xc, labels)
            if input.clustering_method() == "km":
                return textwrap.dedent(f"""\
                Number of clusters: {len(set(labels))}

                Silhouette Score: {sil_score:.4f} (higher is better)
                Inertia: {km_model().inertia_:.4f} (isolated value not meaningful, look for elbow)
                -----------
                """)
            else:
                return textwrap.dedent(f"""\
                Number of clusters: {len(set(labels))}

                Silhouette Score: {sil_score:.4f} (higher is better)
                Inertia: N/A (not defined for hierarchical clustering)
                -----------
                """)


    @output
    @render_plotly
    @reactive.event(input.run_clustering)
    def silhouette():
        labels = clustering()
        Xc, _, _, _ = _prepare_spaces()
        if labels is None or len(set(labels)) < 2:
            return _empty_fig("At least two clusters are required to compute a silhouette plot.", h=600)
        return silhouette_plot(data=Xc, groups=labels)

        
    @reactive.calc
    def segmentation_plot_img() -> Optional[np.ndarray]:
        if input.image_file() is None:
            return None
        else:
            fileinfo = input.image_file()
            img_bytes = fileinfo[0]['datapath']
            img = plt.imread(img_bytes)
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            elif img.shape[2] == 4:
                img = img[:, :, :3]
            return img


    @output
    @render.plot
    @reactive.event(input.run_segmentation)
    def segmentation_plot():
        if segmentation_plot_img() is None:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Upload an image and run segmentation", ha='center', va='center', fontsize=12)
            ax.axis('off')
            return fig
        else:
            img = segmentation_plot_img()
            X = img.reshape(-1, 3)
            k = input.n_segments()
            model = KMeans(n_clusters=k, random_state=input.seed())
            labels = model.fit_predict(X)
            segmented_img = model.cluster_centers_[labels].reshape(img.shape).astype(np.uint8)

            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(img)
            ax[0].set_title("Original Image")
            ax[0].axis('off')
            ax[1].imshow(segmented_img)
            ax[1].set_title(f"Segmented Image (k={k})")
            ax[1].axis('off')
            plt.tight_layout()
            return fig


    @output
    @render_plotly
    @reactive.event(input.run_clustering)
    def linkage_plot():
        Xc, _, _, _ = _prepare_spaces()
        linked = linkage(Xc, method=input.linkage())
        linkage_data.set(linked)
        Z = linkage_data()
        if Z is None:
            return go.Figure().add_annotation(
                text="Run hierarchical clustering to see the linkage (dendrogram).",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )
        return dendrogram_plot(Z, figsize=(7, 6), orientation="top",
                               title="Dendrogram", xaxis_title="Samples",
                               yaxis_title="Euclidean distances")


    @output
    @render.data_frame
    def linkage_info():
        Z = linkage_data()
        if Z is None:
            return pd.DataFrame({"Info": ["Run hierarchical clustering to see the linkage matrix."]})

        df = pd.DataFrame(Z, columns=["Cluster 1", "Cluster 2", "Distance", "Sample Count"])
        df["Cluster 1"] = df["Cluster 1"].astype(int)
        df["Cluster 2"] = df["Cluster 2"].astype(int)
        return df


    @output
    @render_plotly
    def elbow_plot():
        if km_model() is None:
            return go.Figure().add_annotation(
                text="Run K-Means clustering to see the elbow plot.",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            ).update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20))

        Xc, _, _, _ = _prepare_spaces()
        kmax = input.elbow_kmax()

        ks = list(range(2, kmax + 1))
        vals = []
        for k in ks:
            model = KMeans(n_clusters=k, init=input.init(), random_state=input.seed())
            model.fit(Xc)
            vals.append(model.inertia_)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ks, y=vals, mode="lines+markers", name="Inertia"
        ))
        fig.update_layout(
            title="Elbow Method for Optimal k (optimal k = elbow in the curve)",
            xaxis_title="Number of clusters k",
            yaxis_title="Inertia",
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            hovermode="x unified",
        )
        fig.update_xaxes(showline=True, linewidth=2, linecolor="black", mirror=True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor="black", mirror=True)
        return fig

    
    @output
    @render_plotly
    def aic_bic_plot():
        gm = gmm_model()
        if gm is None:
            return go.Figure().add_annotation(
                text="Run GMM clustering to see the AIC/BIC plot.",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            ).update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20))

        Xc, _, _, _ = _prepare_spaces()
        kmax = input.aic_bic_kmax()

        ks = list(range(1, kmax + 1))
        aic_vals, bic_vals = [], []
        for k in ks:
            model = GaussianMixture(
                n_components=k,
                covariance_type=input.covariance_type(),
                random_state=input.seed()
            ).fit(Xc)
            aic_vals.append(model.aic(Xc))
            bic_vals.append(model.bic(Xc))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ks, y=aic_vals, mode="lines+markers", name="AIC"))
        fig.add_trace(go.Scatter(x=ks, y=bic_vals, mode="lines+markers", name="BIC"))

        fig.update_layout(
            title="AIC and BIC for GMM (optimal k = min)",
            xaxis_title="Number of components k",
            yaxis_title="AIC / BIC",
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(title=None, orientation="h", x=0.5, xanchor="center", y=1.08),
            hovermode="x unified",
        )
        fig.update_xaxes(showline=True, linewidth=2, linecolor="black", mirror=True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor="black", mirror=True)
        return fig


    @output
    @render_plotly
    def dbscan_knn_plot():
        Xc, _, _, _ = _prepare_spaces()
        if Xc is None or Xc.size == 0:
            return go.Figure().add_annotation(
                text="Generate data to see the k-NN distance plot for DBSCAN.",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            ).update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20))

        k = input.dbscan_k()
        nbrs = NearestNeighbors(n_neighbors=k).fit(Xc)
        distances, _ = nbrs.kneighbors(Xc)
        k_distances = np.sort(distances[:, k - 1])
        x_idx = np.arange(1, len(k_distances) + 1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_idx, y=k_distances, mode="lines",
            name=f"Distance to {k}-th NN"
        ))
        fig.update_layout(
            title="k-NN Distance Plot (use this to choose epsilon for DBSCAN)",
            xaxis_title="Points sorted by distance to k-th nearest neighbor",
            yaxis_title=f"Distance to {k}-th nearest neighbor",
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
            showlegend=False
        )
        fig.update_xaxes(showline=True, linewidth=2, linecolor="black", mirror=True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor="black", mirror=True)
        return fig


    @output
    @render.data_frame
    @reactive.event(input.generate_data)
    def data_table():
        X, y = data()
        names = feature_names() or [f"Feature {i+1}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=names)
        df["True Label"] = y

        if df is None:
            return pd.DataFrame({"Info": ["No data available."]})
        return df
    

    @output
    @render.data_frame
    @reactive.event(input.run_clustering, input.sample_cluster_nr, input.sample_per_cluster, input.auxiliary_csv)
    def cluster_samples():

        labels = clustering()
        if labels is None or len(np.unique(labels)) == 0:
            return pd.DataFrame({"Info": ["Run clustering first."]})
        
        cluster = int(input.sample_cluster_nr() or 0)
        n_req = int(input.sample_per_cluster() or 1)

        if input.data_kind() == "tweets":
            af = "./resources/bluesky_posts.csv"
        elif input.data_kind() == "spotify":
            af = "./resources/spotify_metadata.csv"
        else:
            auxiliary_file = input.auxiliary_csv()
            af = auxiliary_file[0]['datapath'] if auxiliary_file is not None else None
        
        if af is not None:
            try:
                if af.endswith(".csv"):
                    d = pd.read_csv(af)
                elif af.endswith(".txt"):
                    with open(af, 'r') as f:
                        lines = f.readlines()
                    d = pd.DataFrame({lines[0].strip(): [line.strip() for line in lines[1:]]})
                else:
                    return pd.DataFrame({"Info": ["Unsupported file format."]})
            except Exception as e:
                return pd.DataFrame({"Info": [f"Error reading auxiliary file: {e}"]})
            if d.shape[0] != len(labels):
                return pd.DataFrame({"Info": ["Auxiliary file row count does not match number of samples."]})
        else:
            X, _ = data()
            names = feature_names() or [f"Feature {i+1}" for i in range(X.shape[1])]
            d = pd.DataFrame(X, columns=names)
        
        d["Cluster"] = labels.astype(int)
        subset = d[d["Cluster"] == cluster]
        if subset.empty:
            return pd.DataFrame({"Info": [f"No samples in cluster {cluster}."]})

        n = min(n_req, len(subset))
        if n < n_req:
            ui.notification_show(
                f"Cluster {cluster} has only {n} samples. Showing all available samples.",
                type="warning"
            )
        return subset.sample(n=n, random_state=input.seed()).reset_index(drop=True)


    @output
    @render.data_frame
    def cluster_sizes():
        labels = clustering()
        if labels is None or len(np.unique(labels)) == 0:
            return pd.DataFrame({"Info": ["Run clustering first."]})

        cluster_counts = pd.Series(labels).value_counts().reset_index()
        cluster_counts.columns = ["Cluster", "Size"]
        return cluster_counts


    @output
    @render.data_frame
    def cluster_labels():
        labels = clustering()
        if labels is None or len(np.unique(labels)) == 0:
            return pd.DataFrame({"Info": ["Run clustering first."]})

        df = pd.DataFrame({
            "Sample Index": np.arange(len(labels)),
            "Assigned Cluster": labels.astype(int)
        })
        return df
    

    @output
    @render.text
    @reactive.event(input.generate_data)
    def data_summary():
        X, y = data()
        if X is None or X.size == 0:
            return "No data loaded."
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y)) if y is not None else "N/A"
        return textwrap.dedent(f"""\
        Number of samples: {n_samples}
        Number of features: {n_features}
        Number of true classes (if available): {n_classes if n_classes != 1 else "N/A"}
        """)
    
    


app = App(app_ui, server, static_assets=Path(__file__).parent / "www")

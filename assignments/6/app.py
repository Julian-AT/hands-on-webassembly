from shiny import App, ui, render, reactive
import json, math, numpy as np, pandas as pd, torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import random

try:
    from scipy.special import expit as sigmoid
except Exception:
    def sigmoid(x):
        x = np.asarray(x, dtype=float)
        return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

import u6_utils as u6 

sns.set_theme(style="whitegrid", palette="muted")


# -----------------------------
# Helpers for Architecture parsing/building
# -----------------------------

ACTIVATIONS = {
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
}


def parse_architecture(text: str):
    if not text.strip():
        return []
    data = json.loads(text) if text.strip().lstrip().startswith("{") else json.loads(text)
    layers = data.get("layers", data if isinstance(data, list) else [])
    if not isinstance(layers, list):
        raise ValueError("Architecture must be a list under 'layers' or a list itself.")
    out = []
    for it in layers:
        if not isinstance(it, dict) or "type" not in it:
            raise ValueError("Each layer entry must be an object with at least a 'type' key.")
        t = str(it["type"]).lower()
        p = {k:v for k,v in it.items() if k!="type"}
        if t not in {"flatten","linear","dropout","batchnorm1d", *ACTIVATIONS.keys()}:
            raise ValueError(f"Unsupported layer type for Unit 6: {t}")
        out.append((t, p))
    return out

def build_model(specs, input_shape):
    """
    FC routing:
    - Everything before/after a Flatten lands in an nn.Sequential.
    - We auto-insert Flatten if a Linear appears but no Flatten was specified.
    - Linear: out_features (required); in_features optional ('auto' uses current size).
    """
    layers = []
    c,h,w = input_shape if len(input_shape)==3 else (1, *input_shape) if len(input_shape)==2 else (1,1,input_shape[0])
    feat_in = c*h*w
    seen_flatten = False

    def ensure_flatten():
        nonlocal seen_flatten
        if not seen_flatten:
            layers.append(nn.Flatten())
            seen_flatten = True

    for t, p in specs:
        if t == "flatten":
            layers.append(nn.Flatten()); seen_flatten = True
        elif t == "dropout":
            layers.append(nn.Dropout(float(p.get("p", 0.5))))
        elif t == "batchnorm1d":
            ensure_flatten()
            layers.append(nn.BatchNorm1d(feat_in))
        elif t in ACTIVATIONS:
            layers.append(ACTIVATIONS[t]())
        elif t == "linear":
            ensure_flatten()
            out_f = int(p.get("out_features"))
            in_f  = p.get("in_features","auto")
            if in_f == "auto": in_f = feat_in
            layers.append(nn.Linear(int(in_f), out_f))
            feat_in = out_f
        else:
            raise RuntimeError("Internal: unsupported layer in FC builder.")
    # tiny default head in case of empty spec
    if not layers:
        layers = [nn.Flatten(), nn.Linear(feat_in, 10)]
    return nn.Sequential(*layers)

# -----------------------------
# Presets (simple FCN variants)
# -----------------------------
PRESETS = {
  "MNIST – 2×Hidden (ReLU)": {
    "layers": [
      {"type":"flatten"},
      {"type":"linear", "out_features":128},
      {"type":"relu"},
      {"type":"linear", "out_features":64},
      {"type":"relu"},
      {"type":"linear", "out_features":10}
    ]
  },
  "MNIST – Logistic Regression": {
    "layers": [
      {"type":"flatten"},
      {"type":"linear","out_features":10}
    ]
  },
  "Toy Classification (Binary) – 2×Hidden (Tanh)": {
    "layers": [
      {"type":"flatten"},
      {"type":"linear","out_features":16},
      {"type":"tanh"},
      {"type":"linear","out_features":8},
      {"type":"tanh"},
      {"type":"linear","out_features":2}
    ]
  },
  "Toy Classification (Binary) – 2×Hidden (ReLU)": {
    "layers": [
      {"type":"flatten"},
      {"type":"linear","out_features":16},
      {"type":"relu"},
      {"type":"linear","out_features":8},
      {"type":"relu"},
      {"type":"linear","out_features":2}
    ]
  },
  "Toy Regression – 2×Hidden (ReLU)": {
    "layers": [
      {"type":"flatten"},
      {"type":"linear","out_features":32},
      {"type":"relu"},
      {"type":"linear","out_features":16},
      {"type":"relu"},
      {"type":"linear","out_features":1}
    ]
  },
  "Toy Classification (Binary) – 1×Linear": {
    "layers": [
      {"type": "flatten"},
      {"type": "linear", "out_features": 2}
    ]
  },
  "Toy Regression – 1xLinear": {
    "layers": [
      {"type":"flatten"},
      {"type":"linear", "out_features": 1},
    ]
  }
}

# -----------------------------
# Helper functions for the linear/logistic regression demos
# -----------------------------

def model_linear(x, d, k):
    return d + k * x

def model_logistic(x, d, k):
    logits = model_linear(x, d, k)
    return sigmoid(logits)

def loss_linear(dataset, d, k):
    predictions = model_linear(dataset.x.values, d, k)
    targets = dataset.y.values
    return np.mean((targets - predictions) ** 2, axis=-1)

def loss_logistic(dataset, d, k):
    predictions_before_sigmoid = d + k * dataset.x.values
    targets = dataset.y.values
    return np.mean(np.maximum(predictions_before_sigmoid, 0)
                   - targets * predictions_before_sigmoid
                   + np.log1p(np.exp(-abs(predictions_before_sigmoid))),
                   axis=-1)

def loss_grad_logistic(dataset, d, k):
    predictions = model_logistic(dataset.x.values, d, k)
    targets = dataset.y.values
    delta = predictions - targets
    d_grad = np.mean(delta, axis=-1)
    k_grad = np.mean(dataset.x.values * delta, axis=-1)
    return d_grad, k_grad


# -----------------------------
# UI
# -----------------------------

ace_head = ui.head_content(
    ui.tags.script(src="https://cdnjs.cloudflare.com/ajax/libs/ace/1.32.3/ace.js"),
    ui.tags.style("""
    #arch_editor {
    height: 420px;
    width: 100%;
    border: 1px solid #ddd;
    border-radius: 8px;
    font-size: inherit;
    }
    textarea#arch_text { 
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; 
    }
    """),
    ui.tags.script("""
    (function () {
    var editor;
    var mounted = false;

    function waitForAce(cb) {
        if (window.ace) cb();
        else setTimeout(function(){ waitForAce(cb); }, 50);
    }

    function mount() {
        if (mounted) return;
        var el = document.getElementById('arch_editor');
        if (!el) return;

        editor = ace.edit(el);
        mounted = true;
        window.archAce = editor;
        editor.setTheme('ace/theme/github');
        editor.session.setMode('ace/mode/json');
        editor.setOptions({ wrap:true, showPrintMargin:false, fontSize:'14px', tabSize:2, useSoftTabs:true });

        // start empty; content only comes via set_arch_text message
        editor.session.setValue('', -1);
        editor.clearSelection();

        // keep textarea + Shiny input in sync
        editor.session.on('change', function () {
        var val = editor.getValue();
        var ta2 = document.getElementById('arch_text');
        if (ta2) ta2.value = val;
        if (window.Shiny && Shiny.setInputValue) {
            Shiny.setInputValue('arch_text', val, {priority:'event'});
        }
        });
    }

    function ensureMounted() { waitForAce(mount); }
    function resize() { if (mounted && editor) editor.resize(true); }

    // Resize when the element becomes visible (no Bootstrap dependency)
    function observeVisibilityAndSize() {
        var el = document.getElementById('arch_editor');
        if (!el) return;

        // 1) IntersectionObserver: fires when the element is actually visible
        try {
        var io = new IntersectionObserver(function (entries) {
            entries.forEach(function (e) {
            if (e.isIntersecting) {
                ensureMounted();
                // small delay to let layout settle
                setTimeout(resize, 20);
            }
            });
        }, { root: null, threshold: 0.01 });
        io.observe(el);
        } catch (_) {
        // ignore if not supported
        }

        // 2) ResizeObserver: resizes Ace whenever container size changes
        try {
        var ro = new ResizeObserver(function () {
            resize();
        });
        ro.observe(el);
        } catch (_) {
        // ignore if not supported
        }
    }

    // Fallback: light polling for first 2 seconds to catch first reveal
    function earlyResizePulse() {
        var t0 = Date.now();
        (function pulse(){
        resize();
        if (Date.now() - t0 < 2000) setTimeout(pulse, 120);
        })();
    }

    // Mount once DOM is ready
    document.addEventListener('DOMContentLoaded', function () {
        var ta = document.getElementById('arch_text');
        if (ta) ta.setAttribute('autocomplete','off');

        setTimeout(function () {
        ensureMounted();
        observeVisibilityAndSize();
        earlyResizePulse();
        }, 100);
    });

    // If Bootstrap is present, this still helps; if not, no problem.
    document.addEventListener('shown.bs.tab', function () {
        ensureMounted();
        setTimeout(resize, 10);
    }, true);

    // Also mount/resize when the preset button is clicked (belt & suspenders)
    document.addEventListener('click', function (ev) {
        var t = ev.target;
        if (!t) return;
        if (t.id === 'load_preset' || t.closest && t.closest('#load_preset')) {
        ensureMounted();
        setTimeout(resize, 10);
        }
    }, true);

    // Robust preset receiver — update Ace, textarea, and Shiny input
    (function wirePreset(){
        if (!(window.Shiny && Shiny.addCustomMessageHandler)) {
        setTimeout(wirePreset, 50);
        return;
        }
        Shiny.addCustomMessageHandler('set_arch_text', function (msg) {
        var val = (msg && typeof msg.value === 'string') ? msg.value : '';
        function apply() {
            if (!mounted || !editor) { ensureMounted(); setTimeout(apply, 50); return; }
            editor.session.setValue(val, -1);
            editor.clearSelection();
            var ta = document.getElementById('arch_text');
            if (ta) ta.value = val;
            if (window.Shiny && Shiny.setInputValue) {
            Shiny.setInputValue('arch_text', val, {priority:'event'});
            }
            // after setting content, make sure it's visible-sized
            setTimeout(resize, 10);
        }
        apply();
        });
    })();

    // Window resize is always safe
    window.addEventListener('resize', resize);
    })();
    """),


    # --- Other head content ---
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
        /* Unified table look for render.table */
        table.dataframe, .shiny-table {
            border-collapse: collapse;
            width: auto;                 /* more compact look */
        }
        table.dataframe th, table.dataframe td,
        .shiny-table th, .shiny-table td {
            border: 1px solid #ccc;      /* full grid (vertical + horizontal) */
            padding: 4px 8px;
        }
        /* Centered content */
        table.dataframe th, .shiny-table th {
            text-align: center;
        }
        table.dataframe td, .shiny-table td {
            text-align: center;
        }
    """),


    ui.tags.style("""
    .plot-toolbar {
        background: rgba(255,255,255,.8);
        padding: .5rem .75rem;
        border-radius: .5rem;
        box-shadow: 0 4px 14px rgba(0,0,0,.12);
    }
    .plot-toolbar .form-label { margin-bottom: .25rem; font-size: .875rem; }
    .plot-toolbar .form-control { height: calc(1.6rem + 2px); padding: .125rem .5rem; }
    .plot-frame { min-height: 560px; } /* ensures header+plot occupy same vertical space */
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
    ace_head,
    ui.h2("Hands-on AI I - Unit 6: Your First Neural Networks", class_="text-center mb-3"),
    ui.navset_tab(
        ui.nav_panel("Linear Regression",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h4("Parameters"),
                    ui.input_numeric("linear_reg_k", "Slope (k)", 0.422),
                    ui.input_numeric("linear_reg_d", "Intercept (d)", 0.241),
                    ui.input_numeric("linear_seed", "Random seed", 42),
                    ui.input_slider("linear_n", "Number of samples", 20, 1000, 100, step=10),
                    ui.input_slider("linear_noise", "Variance", 0.0, 2.0, 0.3, step=0.05),
                    ui.input_action_button("make_linear_data", "Generate", class_="btn-primary")
                ),
                ui.navset_card_pill(
                    ui.nav_panel("Plot",
                        ui.output_plot("linear_data", height="520px")                
                    ),
                    ui.nav_panel("Loss",
                        ui.layout_columns(
                            ui.input_slider("linear_loss_d", "d", -10.0, 10.0, [-1, 1], step=0.01),
                            ui.input_slider("linear_loss_k", "k", -10.0, 10.0, [-1, 1], step=0.01),
                            ui.input_slider("linear_loss_grid", "Grid Points", 101, 501, 101),
                        ),
                        ui.output_plot("linear_loss_landscape", height="520px"),
                    ),
                    ui.nav_panel("Parameter Tweaking",
                        ui.output_plot("linear_fitted_model_custom", height="520px"),
                        ui.panel_fixed(
                            ui.div({"class": "plot-toolbar d-flex gap-2 align-items-end"},
                                ui.input_numeric("linear_custom_k", "k", 0.000, step=0.1, width="110px"),
                                ui.input_numeric("linear_custom_d", "d", 0.000, step=0.1, width="110px"),
                            ),
                        ),
                    ),
                    ui.nav_panel("Model Fitting",
                        ui.output_plot("linear_fitted_model_polyfit", height="520px")

                    ),
                ),
            ),   
        ),
        ui.nav_panel("Logistic Regression",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h4("Parameters"),
                    ui.input_numeric("logistic_seed", "Random seed", 42),
                    ui.input_slider("logistic_n", "Number of samples", 20, 1000, 100, step=10),
                    ui.input_slider("logistic_noise", "Variance (σ)", 0.0, 2.0, 0.3, step=0.05),
                    ui.input_slider("logistic_thr", "Threshold (for labels)", 0.0, 1.0, 0.5, step=0.01),
                    ui.input_action_button("make_logistic_data", "Generate", class_="btn-primary")
                ),
                ui.navset_card_pill(
                    ui.nav_panel("Plot",
                        ui.output_plot("logistic_data", height="520px")                
                    ),
                    ui.nav_panel("Loss",
                        ui.layout_columns(
                            ui.input_slider("logistic_loss_d", "d", -10.0, 10.0, [-10, 10], step=0.01),
                            ui.input_slider("logistic_loss_k", "k", -10.0, 10.0, [-10, 10], step=0.01),
                            ui.input_slider("logistic_loss_grid", "Grid Points", 101, 501, 101),
                        ),
                        ui.output_plot("logistic_loss_landscape", height="520px"),
                        ui.input_checkbox("show_gradient", "Show Gradient"),
                    ),
                    ui.nav_panel("Parameter Tweaking",
                        ui.output_plot("logistic_fitted_model_custom", height="520px"),
                        ui.panel_fixed(
                            ui.div({"class": "plot-toolbar d-flex gap-2 align-items-end"},
                                ui.input_numeric("logistic_custom_k", "k", 0.000, step=0.1, width="110px"),
                                ui.input_numeric("logistic_custom_d", "d", 0.000, step=0.1, width="110px"),
                            ),
                        ),
                    ),
                    ui.nav_panel("Gradient Descent",
                        
                        ui.layout_column_wrap(
                            ui.div(ui.input_numeric("sgd_start_d", "Start d", 7.5, step=0.1), class_="mb-2"),
                            ui.div(ui.input_numeric("sgd_start_k", "Start k", -6.0, step=0.1), class_="mb-2"),
                            ui.div(ui.input_numeric("sgd_iterations", "Iterations", 1000, step=10), class_="mb-2"),
                            ui.div(ui.input_numeric("sgd_learning_rate", "Learning Rate", 0.01, step=0.01), class_="mb-2"),
                            ui.div(ui.input_numeric("sgd_momentum", "Momentum", 0.9, step=0.01), class_="mb-2"),
                        ),
                        ui.input_action_button("do_gradient", "Run Gradient Descent", class_="btn-success mt-3"),
                        ui.div(class_="my-3"),
                        ui.div(ui.output_text_verbatim("sgd_result"), class_="mb-2"),
                        ui.output_plot("logistic_gradient", height="520px"),
                        
                    ),
                    ui.nav_panel("Fitted Model using GD",
                        ui.output_plot("logistic_fitted_model", height="520px")
                    ),
                ),
            ),   
        ),
        ui.nav_panel("FNN: Data",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_select(
                        "dataset", "Dataset",
                        {
                          "toy_reg":"Toy Regression (y = a + b x + noise)",
                          "toy_sine": "Toy Regression (sine wave + noise)",
                          "toy_bin":"Toy Classification (threshold + noise)",
                          "toy_bin2": "Toy Classification (double interval)",
                          "blob2d": "Toy 2D Blobs (nonlinear classification)",
                          "moons2d": "Toy 2D Moons (nonlinear classification)",
                          "MNIST":"MNIST (10 classes)",
                          "FashionMNIST":"FashionMNIST (10 classes)"
                        }, selected="MNIST"
                    ),
                    ui.input_numeric("seed", "Random seed", 42),
                    ui.hr(),
                    ui.panel_conditional("input.dataset === 'toy_reg' || input.dataset === 'toy_sine'",
                        ui.input_slider("n_pairs","Num. samples", 20, 2000, 200, step=20),
                        ui.input_slider("variance","Noise (σ)", 0.0, 2.0, 0.3, step=0.05),
                        ui.panel_conditional("input.dataset === 'toy_reg'",
                            ui.input_text("coeffs","Coefficients (comma-sep)", "0.241,0.422"),
                        ),
                    ),
                    ui.panel_conditional("input.dataset === 'blob2d' || input.dataset === 'moons2d'",
                        ui.input_slider("blob_n", "Num. samples", 50, 2000, 250, step=50),
                        ui.input_slider("blob_var", "Noise (σ)", 0.01, 2.0, 0.25, step=0.01),
                        ui.panel_conditional("input.dataset === 'blob2d'",
                            ui.input_slider("blob_thr", "Threshold (radius)", 0.1, 3.0, 1.25, step=0.05),
                            ui.input_text("blob_offset", "Offset (x,y)", "0.25, 0.25"),
                        ),
                    ),
                    ui.panel_conditional("input.dataset === 'toy_bin' || input.dataset === 'toy_bin2'",
                        ui.input_slider("n_pairs_c","Num. samples", 20, 5000, 500, step=20),
                        ui.input_slider("variance_c","Noise (σ)", 0.0, 0.5, 0.1, step=0.01),
                        ui.panel_conditional("input.dataset === 'toy_bin'",
                            ui.input_slider("threshold","Threshold", 0.0, 1.0, 0.5, step=0.01),
                        ),
                    ),
                    ui.panel_conditional("(input.dataset === 'MNIST') || (input.dataset === 'FashionMNIST')",
                        ui.input_slider("batch","Batch size", 4, 256, 64, step=4),
                        ui.input_slider("valid","Validation split", 0.0, 0.5, 0.1, step=0.05),
                    ),
                    ui.hr(),
                    ui.input_action_button("load","Load/Reload data", class_="btn-success")
                ),
                ui.panel_conditional("input.load > 0",
                    ui.card(
                        ui.card_header("Dataset Information"),
                        ui.output_text_verbatim("data_info"),
                    ),
                    ui.card(
                        ui.card_header("Sample Visualization"),
                        ui.panel_conditional("input.dataset === 'MNIST' || input.dataset === 'FashionMNIST'",
                            ui.input_numeric("sample_ix","Sample index", 0, min=0, step=1),
                        ),
                        ui.output_plot("sample_plot", height="520px"),
                    )
                )
            )
        ),
        ui.nav_panel("FNN:Architecture",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_select("preset","Preset", {k:k for k in PRESETS.keys()}),
                    ui.input_action_button("load_preset","Load preset"),
                    ui.hr(),
                    ui.input_action_button("apply_arch","Apply architecture", class_="btn-success"),
                    ui.download_button("export_arch","Export JSON", class_="btn-primary"),
                ),
                ui.card(
                    ui.card_header("Architecture Editor"),
                    ui.tags.p("Edit architecture JSON (Unit 6 supports: flatten, linear, dropout, batchnorm1d, relu/leakyrelu/sigmoid/tanh/gelu)."),
                    ui.div(ui.input_text_area("arch_text", None, rows=1, width="100%", value=""), style="display:none;"),
                    ui.tags.div(id="arch_editor"),
                ),
                ui.card(
                    ui.card_header("Architecture Table"),
                    ui.output_ui("arch_table_block"),
                ),
                ui.card(
                    ui.card_header("Model Architecture (PyTorch)"),
                    ui.output_text_verbatim("model_summary"),
                ),            
            )
        ),
        ui.nav_panel("FNN: Training",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_slider("epochs","Epochs", 1, 100, 5),
                    ui.input_numeric("lr","Learning rate", 0.01, step=0.001),
                    ui.input_numeric("momentum","Momentum", 0.9, step=0.01, min=0.0, max=0.99),
                    # disabling CUDA option for reproducibility
                    #ui.input_checkbox("cuda","Use CUDA if available", False),
                    ui.output_text_verbatim("which_device"),
                    ui.input_checkbox("early","Early stopping (val loss)", False),
                    ui.input_numeric("patience","Patience (epochs)", 3),
                    ui.input_numeric("train_seed","Training seed", 42),
                    ui.input_action_button("train","Train model", class_="btn-success"),
                    ui.input_action_button("reset","Reset", class_="btn-danger"),
                ),
                ui.card(
                    ui.card_header("Loss Plot"),
                    ui.output_plot("loss_plot"),
                ),
                ui.card(
                    ui.card_header("Training Progress"),
                    ui.output_ui("train_progress"),
                ),
                ui.card(
                    ui.card_header("Best Model Info"),
                    ui.output_text_verbatim("best_model_info"),
                ),
                ui.card(
		    ui.card_header("Parameter Values"),
		    ui.output_text_verbatim("param_values"),
		),
            )
        ),
        ui.nav_panel("FNN: Predict",
            ui.tags.div(style="height: 20px;"),
            ui.div(
                ui.card(
                    ui.card_header("Predictions"),
                    ui.output_plot("pred_plot", height="520px"),
                    ui.panel_conditional("input.dataset === 'MNIST' || input.dataset === 'FashionMNIST'",
                        ui.output_text_verbatim("metrics_text"),
                    ),
                style="margin-left: 40px; margin-right: 40px;"
                )
            )
        ),
        ui.nav_panel("FNN: Analysis",
            ui.tags.div(style="height: 20px;"),
            ui.panel_conditional("input.dataset === 'MNIST' || input.dataset === 'FashionMNIST'",
                ui.div(
                    ui.navset_card_pill(
                        ui.nav_panel("Confusion Matrix",
                            ui.input_checkbox("cm_norm","Normalize confusion matrix", True),
                            ui.output_plot("conf_mat_plot", height="520px"),
                        ),
                        ui.nav_panel("Input Weights",
                            ui.output_plot("input_weights_plot", height="520px")
                        ),
                    ), style="margin-left: 40px; margin-right: 40px;"
                )
            ),
            ui.panel_conditional("input.dataset !== 'MNIST' && input.dataset !== 'FashionMNIST'",
                ui.card(
                    ui.card_header("Note"),
                    ui.p("Analysis not available for this dataset."),
                    style="margin-left: 15px; margin-right: 40px;"
                )
            )
        )
    )
)

# -----------------------------
# Server
# -----------------------------
def server(input, output, session):

    # Tabs 1 and 2: reactive calculations for the data

    @reactive.calc
    def make_linear_data():
        np.random.seed(int(input.linear_seed() or 42))
        n = int(input.linear_n())
        noise = float(input.linear_noise())
        coeff_k = float(input.linear_reg_k())
        coeff_d = float(input.linear_reg_d())
        df = u6.get_dataset(num_pairs=n, variance=noise, coefficients=(coeff_d, coeff_k))
        return df
    
    @reactive.calc
    def make_logistic_data():
        np.random.seed(int(input.logistic_seed() or 42))
        n = int(input.logistic_n())
        noise = float(input.logistic_noise())
        df = u6.get_dataset_logistic(n, threshold=float(input.logistic_thr()), variance=noise)
        return df
    
    # Tabs 1 and 2: reactive calculation for the plot titles

    @reactive.calc
    def plot_title_linear():
        df = make_linear_data()
        return f"Linear Regression: g(x;d,k) = {float(input.linear_reg_d())} + {float(input.linear_reg_k())} * x"

    @reactive.calc
    def plot_title_logistic():
        df = make_logistic_data()
        pos = int(df['y'].sum())
        return f"Logistic Regression: N = {len(df)}, positives = {pos}"

    @output
    @render.plot
    @reactive.event(input.make_linear_data)
    def linear_data():
        fig, ax = plt.subplots()
        df = make_linear_data()
        sns.scatterplot(data=df, x="x", y="y", color="black", edgecolor=None, s=30)

        ax.set_title(plot_title_linear())
        return fig

    @output
    @render.plot
    @reactive.event(input.make_logistic_data)
    def logistic_data():
        fig, ax = plt.subplots()
        df = make_logistic_data()
        sns.scatterplot(data=df, x="x", y="y", hue="y", palette="viridis", ax=ax)
        ax.set_title(plot_title_logistic())
        return fig
    

    # Tab 1 and 2: loss landscapes

    @output
    @render.plot
    def linear_loss_landscape():
        df = make_linear_data()
        d_values = np.linspace(input.linear_loss_d()[0], input.linear_loss_d()[1], input.linear_loss_grid())
        k_values = np.linspace(input.linear_loss_k()[0], input.linear_loss_k()[1], input.linear_loss_grid())
        u6.plot_loss_landscape(loss_linear, df, d=d_values, k=k_values)
        plt.title("Linear Loss Landscape")
        return plt.gcf()

    
    @reactive.calc
    def logistic_loss_landscape_calc():
        df = make_logistic_data()
        d_values = np.linspace(input.logistic_loss_d()[0], input.logistic_loss_d()[1], input.logistic_loss_grid())
        k_values = np.linspace(input.logistic_loss_k()[0], input.logistic_loss_k()[1], input.logistic_loss_grid())
        u6.plot_loss_landscape(loss_logistic, df, d=d_values, k=k_values)
        if input.show_gradient():
            u6.plot_loss_landscape(loss_logistic, df, d=d_values, k=k_values, grad_fn=loss_grad_logistic)
        plt.title("Logistic Loss Landscape")
        return plt.gcf()
    
    @output
    @render.plot
    def logistic_loss_landscape():
        return logistic_loss_landscape_calc()

    # Tabs 1 and 2 model fitting

    @reactive.calc
    def lm():
        df = make_linear_data()
        d, k = np.polyfit(x=df.x, y=df.y, deg=1)[::-1]
        return d, k


    @output
    @render.plot
    def linear_fitted_model_custom():
        fig, ax = plt.subplots()
        d = float(input.linear_custom_d())
        k = float(input.linear_custom_k())
        if d == 0 and k == 0:
            ax.text(0.5, 0.5, "No custom model parameters set", ha='center', va='center')
            return fig
        df = make_linear_data()
        sns.scatterplot(data=df, x="x", y="y", color="black", edgecolor=None, s=30)
        
        x = np.linspace(df["x"].min(), df["x"].max(), 200)
        d = float(input.linear_custom_d()); k = float(input.linear_custom_k())
        y = d + k * x
        ax.plot(x, y, color="tab:blue", linewidth=2)

        ax.set_title("Linear Model with k = {:.3f}, d = {:.3f}, loss = {:.3f}".format(k, d, loss_linear(df, d, k)))
        ax.legend().set_visible(False)
        return fig

    @output
    @render.plot
    def linear_fitted_model_polyfit():
        fig, ax = plt.subplots()
        df = make_linear_data()
        d, k = lm()
        sns.scatterplot(data=df, x="x", y="y", color="black", edgecolor=None, s=30)
        x = np.linspace(df["x"].min(), df["x"].max(), 200)
        y = d + k * x
        ax.plot(x, y, color="tab:blue", linewidth=2)

        ax.set_title("Linear Model fitted with np.polyfit with k = {:.3f}, d = {:.3f}, loss = {:.3f}".format(k, d, loss_linear(df, d, k)))
        ax.legend().set_visible(False)
        return fig

    @output
    @render.plot
    def logistic_fitted_model_custom():
        df = make_logistic_data()
        d = float(input.logistic_custom_d())
        k = float(input.logistic_custom_k())
        fig, ax = plt.subplots()
        if d == 0 and k == 0:
            ax.text(0.5, 0.5, "No custom model parameters set", ha='center', va='center')
            return plt.gcf()
        sns.scatterplot(data=df, x="x", y="y", hue="y", palette="viridis")
        x = np.linspace(df["x"].min(), df["x"].max(), 200)
        y = model_logistic(x, d, k)
        ax.plot(x, y, color="tab:blue", linewidth=2)

        ax.set_title("Logistic Model with k = {:.3f}, d = {:.3f}, loss = {:.3f}".format(k, d, loss_logistic(df, d, k)))
        ax.legend().set_visible(False)
        return fig

    @output
    @render.text
    @reactive.event(input.do_gradient)
    def sgd_result():
        df = make_logistic_data()
        d0 = float(input.sgd_start_d())
        k0 = float(input.sgd_start_k())
        lr = float(input.sgd_learning_rate())
        mom = float(input.sgd_momentum())
        steps = int(input.sgd_iterations())
        vd = vk = 0.0
        d, k = d0, k0
        for _ in range(steps):
            g_d, g_k = loss_grad_logistic(df, d, k)
            vd = mom * vd - (1.0 - mom) * g_d
            vk = mom * vk - (1.0 - mom) * g_k
            d += lr * vd
            k += lr * vk

        return (
            f"Initial: d={d0:.3f}, k={k0:.3f}, loss={loss_logistic(df, d0, k0):.3f}\n"
            f"Final:   d={d:.3f}, k={k:.3f}, loss={loss_logistic(df, d, k):.3f}"
        )

    d_sgd, k_sgd = reactive.Value(0.0), reactive.Value(0.0)

    @output
    @render.plot
    @reactive.event(input.do_gradient)
    def logistic_gradient():
        df = make_logistic_data()
        d = float(input.sgd_start_d())
        k = float(input.sgd_start_k())
        lr = float(input.sgd_learning_rate())
        iterations = int(input.sgd_iterations())
        momentum = float(input.sgd_momentum())
        d, k = u6.plot_gradient_descent(
            loss_logistic, loss_grad_logistic, df,
            d=d, k=k,
            steps=iterations,
            stepsize=lr,
            momentum=momentum
        )
        d_sgd.set(d)
        k_sgd.set(k)
        plt.title("Optimal Parameters at k={:.3f}, d={:.3f}".format(k, d))
        return plt.gcf()
    
    @output
    @render.plot
    def logistic_fitted_model():
        fig, ax = plt.subplots()
        df = make_logistic_data()
        d = d_sgd.get()
        k = k_sgd.get()
        if d == 0 and k == 0:
            ax.text(0.5, 0.5, "Run Gradient Descent first", ha='center', va='center')
            return fig
        sns.scatterplot(data=df, x="x", y="y", hue="y", palette="viridis")
        x = np.linspace(df["x"].min(), df["x"].max(), 200)
        y = model_logistic(x, d, k)
        ax.plot(x, y, color="tab:blue", linewidth=2)

        ax.set_title("Logistic Model fitted with GD with k = {:.3f}, d = {:.3f}, loss = {:.3f}".format(k, d, loss_logistic(df, d, k)))
        ax.legend().set_visible(False)
        return fig


    # ----- Data loaders / data frames -----
    @reactive.calc
    @reactive.event(input.load)
    def data_bundle():
        np.random.seed(int(input.seed() or 42))
        ds = input.dataset()

        if ds == "toy_reg":
            coeffs = [float(x.strip()) for x in str(input.coeffs() or "0.241,0.422").split(",")]
            df = u6.get_dataset(int(input.n_pairs()), float(input.variance()), coefficients=coeffs)
            return {"kind": "toy_reg", "df": df, "dataset_name": ds}

        if ds == "toy_sine":
            df = u6.get_dataset_sine(
                num_pairs=int(input.n_pairs()),
                variance=float(input.variance())
            )
            return {"kind": "toy_reg", "df": df, "dataset_name": ds}

        if ds == "toy_bin":
            df = u6.get_dataset_logistic(
                int(input.n_pairs_c()),
                threshold=float(input.threshold()),
                variance=float(input.variance_c())
            )
            return {"kind": "toy_bin", "df": df, "dataset_name": ds}

        if ds == "toy_bin2":
            df = u6.get_dataset_double_interval(
                num_pairs=int(input.n_pairs_c()),
                variance=float(input.variance_c())
            )
            return {"kind": "toy_bin", "df": df, "dataset_name": ds}

        if ds == "blob2d":
            np.random.seed(int(input.seed() or 42))
            off = [float(x.strip()) for x in input.blob_offset().split(",")]
            df = u6.get_dataset_blob2d(
                num_samples=int(input.blob_n()),
                variance=float(input.blob_var()),
                threshold=float(input.blob_thr()),
                offset=tuple(off)
            )
            return {"kind": "blob2d", "df": df, "dataset_name": ds}

        if ds == "moons2d":
            df = u6.get_dataset_moons(
                num_samples=int(input.blob_n()),
                noise=float(input.blob_var())
            )
            return {"kind": "blob2d", "df": df, "dataset_name": ds}

        # image datasets (MNIST / FashionMNIST)
        L = u6.get_dataset_mnist(
            batch_size=int(input.batch()),
            horizontal_flip_p=0.0, vertical_flip_p=0.0, invert=False,
            valid_size=float(input.valid()), augment_train=False, augment_test=False,
            variant="MNIST" if ds=="MNIST" else "FashionMNIST"
        )  # returns (train, [valid], test) 
        if len(L)==2:
            train_loader, test_loader = L
            val_loader = None
        else:
            train_loader, val_loader, test_loader = L
        return {"kind":"img", "train":train_loader, "val":val_loader, "test":test_loader, "variant": ds, "dataset_name": ds}

    # Info + sample preview
    @render.text
    def data_info():
        b = data_bundle()

        if b.get("dataset_name") != input.dataset():
            return (
                "Dataset selection has changed.\n\n"
                "Please click 'Load/Reload data' to generate the new dataset."
            )

        if b["kind"] == "toy_reg":
            df = b["df"]
            if b["dataset_name"] == "toy_sine":
                return f"Sine-wave regression: N={len(df)}, columns={list(df.columns)}"
            else:
                return f"Toy Regression (line): N={len(df)}, columns={list(df.columns)}"

        if b["kind"] == "toy_bin":
            df = b["df"]
            pos = int(df['y'].sum())
            if b["dataset_name"] == "toy_bin2":
                return f"Toy Classification (double interval): N={len(df)}, positives={pos}"
            else:
                return f"Toy Classification (threshold): N={len(df)}, positives={pos}"

        if b["kind"] == "blob2d":
            df = b["df"]
            pos = int(df['y'].sum())
            name = b.get("dataset_name", "blob2d")
            label = {
                "blob2d"   : "Toy 2D Blobs",
                "moons2d"  : "Toy 2D Moons",
            }.get(name, "Toy 2D Blobs")
            return f"{label}: N={len(df)}, positives={pos}"

        # image datasets
        tr = b["train"]; vl = b["val"]; te = b["test"]
        ntr, nte = len(tr.dataset), len(te.dataset)
        nvl = len(vl.dataset) if vl is not None else 0
        try:
            x0, _ = next(iter(tr))
            shape = tuple(x0.shape[1:])
        except StopIteration:
            shape = ("?",)
        return (
            f"Loaded {b['variant']} data.\n\n"
            f"Training set   = {ntr:5d}\nValidation set = {nvl:5d}\nTest set       = {nte:5d}\n\n"
            f"Input shape: {shape}"
        )


    @render.plot
    def sample_plot():
        fig = plt.figure()
        try:
            b = data_bundle()
        except Exception:
            plt.title("Load data first"); return fig
        
        if b.get("dataset_name") != input.dataset():
            plt.title("Dataset selection changed – click 'Load/Reload data' first")
            plt.axis("off")
            return fig

        if b["kind"]=="toy_reg":
            df = b["df"]
            plt.scatter(df["x"], df["y"], s=18)
            plt.xlabel("x"); plt.ylabel("y"); plt.title("Toy Regression samples"); return fig

        if b["kind"]=="toy_bin":
            df = b["df"]
            plt.scatter(df["x"], df["y"], s=18, c=df["y"], cmap="coolwarm")
            plt.xlabel("x"); plt.ylabel("y"); plt.title("Toy Classification samples"); return fig

        if b["kind"] == "blob2d":
            df = b["df"]
            name = b.get("dataset_name", "blob2d")
            label = {
                "blob2d":  "Toy 2D Blobs",
                "moons2d": "Toy 2D Moons",
            }.get(name, "Toy 2D Blobs")
            plt.scatter(df["x1"], df["x2"], s=18, c=df["y"], cmap="coolwarm")
            plt.xlabel("x1"); plt.ylabel("x2")
            plt.title("2D Blob Dataset")
            return fig


        tr = b["train"].dataset
        if len(tr)==0:
            plt.title("Empty train split"); return fig
        ix = max(0, min(int(input.sample_ix() or 0), len(tr)-1))
        x,y = tr[ix]
        x = x.numpy() if isinstance(x, torch.Tensor) else np.array(x)
        if x.ndim==3 and x.shape[0]==1: plt.imshow(x[0], cmap="gray")
        elif x.ndim==3 and x.shape[0]==3: plt.imshow(np.moveaxis(x,0,-1))
        else: plt.imshow(x, cmap="gray")
        plt.xticks([]); plt.yticks([]); plt.title(f"Sample {ix} (label={int(y)})")
        return fig

    # ----- Architecture state + Ace sync -----
    arch_state = reactive.Value("")
    @reactive.effect
    @reactive.event(input.load_preset)
    async def _load_preset():
        name = str(input.preset())
        txt = json.dumps(PRESETS[name], indent=2)
        arch_state.set(txt)
        ui.update_text_area("arch_text", value=txt)
        await session.send_custom_message("set_arch_text", {"value": txt})

    @reactive.effect
    @reactive.event(input.apply_arch)
    def _apply_arch():
        txt = input.arch_text() or arch_state()
        if txt: arch_state.set(txt)

    @render.table
    def arch_table():
        txt = arch_state()
        if not txt.strip(): return pd.DataFrame()
        try:
            specs = parse_architecture(txt)
        except Exception as e:
            return pd.DataFrame({"error":[str(e)]})
        rows=[]
        for i,(t,p) in enumerate(specs):
            row={"#":i,"type":t}; row.update({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in p.items()})
            rows.append(row)
        return pd.DataFrame(rows).replace({np.nan:""})

    @render.ui
    def arch_table_block():
        txt = arch_state()
        try:
            specs = parse_architecture(txt)
            if specs: return ui.output_table("arch_table")
        except Exception: pass
        return ui.div()

    @render.text
    @reactive.event(arch_state)
    def model_summary():
        if not arch_state().strip():
            return "No architecture defined"
        b = data_bundle()

        # Determine input shape
        kind = b["kind"]
        if kind == "toy_reg":
            in_shape = (1,)
        elif kind == "toy_bin":
            in_shape = (1,)
        elif kind == "blob2d":
            in_shape = (2,)
        else:
            x0, _ = next(iter(b["train"]))
            in_shape = tuple(x0.shape[1:])

        try:
            specs = parse_architecture(arch_state())
            model = build_model(specs, in_shape)
            n_params = sum(p.numel() for p in model.parameters())
            return f"{repr(model)}\n\nTotal parameters: {n_params}"
        except Exception as e:
            return f"Model build error: {e}"


    @render.download(filename=lambda: "architecture_u6.json")
    def export_arch():
        txt = arch_state()
        def _w(): yield txt.encode("utf-8")
        return _w()
    

    # ----- Training -----
    trained_model = reactive.Value(None)
    history_df    = reactive.Value(pd.DataFrame())
    best_info     = reactive.Value("")
    train_progress_pct = reactive.Value(0)
    train_progress_msg = reactive.Value("")
    selected_device = reactive.Value("cpu")
    
    @render.ui
    def train_progress():
        pct = int(train_progress_pct())
        msg = train_progress_msg()
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

    @render.text
    def best_model_info():
        return best_info() or "Best model: (not set yet)"

    @reactive.effect
    @reactive.event(input.reset)
    def _reset_training():
        trained_model.set(None)
        history_df.set(pd.DataFrame())
        best_info.set("")
        train_progress_pct.set(0)
        train_progress_msg.set("")

    @render.text
    def which_device():
        return f"Using device: {selected_device.get()}"
    
    @reactive.effect
    @reactive.event(input.train)
    async def _train():

        seed = int(input.train_seed() or 0)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        try:
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Fall back silently if this PyTorch version / backend doesn't support it
            pass

        # Reset progress UI
        train_progress_pct.set(0)
        train_progress_msg.set("Initializing…")
        best_info.set("")
        history_df.set(pd.DataFrame())
        await reactive.flush()

        # Build model
        b = data_bundle()
        specs = parse_architecture(arch_state())
        ds_kind = b["kind"]

        # Determine input shape
        if ds_kind == "toy_reg":
            in_shape = (1,)
        elif ds_kind == "toy_bin":
            in_shape = (1,)
        elif ds_kind == "blob2d":
            in_shape = (2,)
        else:
            x0, _ = next(iter(b["train"]))
            in_shape = tuple(x0.shape[1:])

        model = build_model(specs, in_shape)

        # keep device 'cpu' for reproducibility
        # use_cuda = bool(input.cuda()) and torch.cuda.is_available()
        # selected_device.set(torch.device("cuda" if use_cuda else "cpu"))

        device = selected_device.get()
        model.to(device)

        # Choose loss + loaders per dataset
        ds_kind = b["kind"]
        if ds_kind == "toy_reg":
            loss_fn = nn.MSELoss()
            df = b["df"]
            X = torch.tensor(df["x"].values, dtype=torch.float32).view(-1, 1)
            y = torch.tensor(df["y"].values, dtype=torch.float32).view(-1, 1)
            dset = torch.utils.data.TensorDataset(X, y)
            loader = torch.utils.data.DataLoader(dset, batch_size=64, shuffle=True)
            val_loader = None
        elif ds_kind == "toy_bin":
            loss_fn = nn.CrossEntropyLoss()
            df = b["df"]
            X = torch.tensor(df["x"].values, dtype=torch.float32).view(-1, 1)
            y = torch.tensor(df["y"].values, dtype=torch.long).view(-1)
            dset = torch.utils.data.TensorDataset(X, y)
            loader = torch.utils.data.DataLoader(dset, batch_size=64, shuffle=True)
            val_loader = None
        elif ds_kind == "blob2d":
            loss_fn = nn.CrossEntropyLoss()
            df = b["df"]
            X = torch.tensor(df[["x1", "x2"]].values, dtype=torch.float32)
            y = torch.tensor(df["y"].values, dtype=torch.long)
            dset = torch.utils.data.TensorDataset(X, y)
            loader = torch.utils.data.DataLoader(dset, batch_size=64, shuffle=True)
            val_loader = None
        else:
            loss_fn = nn.CrossEntropyLoss()
            loader = b["train"]
            val_loader = b["val"]

        # Optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=float(input.lr()), momentum=float(input.momentum()))

        epochs = int(input.epochs())
        n_batches = max(1, len(loader))

        hist_rows = []
        best_state = None
        best_epoch = None
        best_val = None
        stopped_early = False
        use_es = bool(input.early())
        patience = int(input.patience() or 0)
        patience_ctr = 0

        # training loop with per-batch UI updates and per-epoch plot refresh
        for ep in range(epochs):
            model.train(True)
            train_sum = 0.0

            for i, (xb, yb) in enumerate(loader):
                xb = xb.to(device)
                yb = yb.to(device)

                preds = model(xb)
                loss = loss_fn(preds, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_sum += loss.item()

                # progress per batch
                done = ep * n_batches + (i + 1)
                total = epochs * n_batches
                pct = int(100 * done / max(1, total))
                train_progress_pct.set(pct)
                train_progress_msg.set(f"Epoch {ep+1}/{epochs} · batch {i+1}/{n_batches}")
                await reactive.flush()

            train_loss = train_sum / n_batches

            # validation (if available)
            row = {"training loss": float(train_loss)}
            if val_loader is not None:
                model.train(False)
                val_sum = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(device); yb = yb.to(device)
                        preds = model(xb)
                        vloss = loss_fn(preds, yb)
                        val_sum += vloss.item()
                val_loss = val_sum / max(1, len(val_loader))
                row["validation loss"] = float(val_loss)

                # early stopping tracking
                if use_es:
                    if best_val is None or val_loss < best_val - 1e-12:
                        best_val = val_loss
                        best_epoch = ep + 1
                        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                        patience_ctr = 0
                    else:
                        patience_ctr += 1
                        if patience_ctr >= patience > 0:
                            stopped_early = True
            hist_rows.append(row)

            # update history df and plot after each epoch
            df_hist = pd.DataFrame(hist_rows, index=range(1, len(hist_rows)+1))
            history_df.set(df_hist)
            await reactive.flush()

            if stopped_early:
                break

        # Restore/escalate best/last info
        if (val_loader is not None) and (best_state is not None):
            model.load_state_dict(best_state)
            best_info.set(f"Using best model from epoch {best_epoch} with validation loss={best_val:.4f}" + (" (stopped early)" if stopped_early else ""))
        else:
            last_ep = len(hist_rows)
            val_part = f", val={hist_rows[-1].get('validation loss'):.4f}" if 'validation loss' in hist_rows[-1] else ""
            best_info.set(f"Using last epoch model (epoch {last_ep}, train={hist_rows[-1]['training loss']:.4f}{val_part})")

        model.eval()
        trained_model.set(model.cpu())
        train_progress_msg.set("Training complete")
        train_progress_pct.set(100)
        await reactive.flush()



    @render.plot
    def loss_plot():
        df = history_df()
        fig = plt.figure()
        if df is None or df.empty:
            plt.title("No training yet"); return fig
        xs = list(range(1, len(df)+1))
        for col in df.columns:
            plt.plot(xs, df[col].values, label=col)
        plt.xlabel("Epoch")
        plt.xticks(xs)
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training curves")
        return fig
    

    # ----- Predict & Analysis -----
    @output
    @render.plot
    def pred_plot():
        fig = plt.figure()
        m = trained_model()
        if m is None: 
            plt.title("Train a model first")
            plt.axis("off")
            return fig
        
        m.eval()
        b = data_bundle()
        n = 9 # fixed number of samples

        if b["kind"].startswith("toy"):
            # Visualize with a simple scatter and model fit curve if 1D
            df = b["df"]
            # Build input grid:
            xx = np.linspace(df["x"].min(), df["x"].max(), 200, dtype=np.float32).reshape(-1,1)
            with torch.no_grad():
                yy = m(torch.from_numpy(xx)).numpy()
            if b["kind"]=="toy_reg":
                plt.scatter(df["x"], df["y"], s=18, alpha=0.6)
                plt.plot(xx.squeeze(), yy.squeeze(), linewidth=2)
                plt.title("Model fit (toy regression)")
            else:
                # Assume 2-logit head - probability of class 1:
                probs = torch.softmax(torch.from_numpy(yy), dim=1).numpy()[:,1]
                plt.scatter(df["x"], df["y"], s=18, c=df["y"], cmap="coolwarm")
                plt.plot(xx.squeeze(), probs, linewidth=2)
                plt.title("Decision curve (toy classification)")
            plt.xlabel("x"); plt.ylabel("y"); return fig
        
        # 2d case
        if b["kind"] == "blob2d":
            df = b["df"]
            X = df[["x1", "x2"]].values.astype(np.float32)

            # grid for decision surface
            x1_min, x1_max = X[:,0].min()-0.5, X[:,0].max()+0.5
            x2_min, x2_max = X[:,1].min()-0.5, X[:,1].max()+0.5
            xx, yy = np.meshgrid(
                np.linspace(x1_min, x1_max, 200),
                np.linspace(x2_min, x2_max, 200)
            )
            grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)

            with torch.no_grad():
                logits = m(torch.from_numpy(grid))
                probs = torch.softmax(logits, dim=1)[:,1].numpy()

            plt.contourf(xx, yy, probs.reshape(xx.shape), levels=20, cmap="coolwarm", alpha=0.6)
            plt.scatter(df["x1"], df["x2"], c=df["y"], cmap="coolwarm", s=20, edgecolor="k")
            plt.xlabel("x1"); plt.ylabel("x2")
            name = b.get("dataset_name", "blob2d")
            label = {
                "blob2d":  "Toy 2D Blobs",
                "moons2d": "Toy 2D Moons",
            }.get(name, "Toy 2D Blobs")

            plt.title(f"Decision boundary ({label})")
            return fig


        # Image case: sample from test loader
        test_loader = b["test"]
        xs,ys = next(iter(test_loader))
        xs,ys = xs[:n], ys[:n]
        with torch.no_grad():
            logits = m(xs); preds = logits.argmax(dim=1)
        # Grid plot
        cols = int(math.ceil(n**0.5)); rows = int(math.ceil(n/cols))
        for i in range(n):
            ax = plt.subplot(rows, cols, i+1)
            x = xs[i].numpy()
            if x.shape[0]==1: ax.imshow(x[0], cmap="gray")
            else: ax.imshow(np.moveaxis(x,0,-1))
            ax.set_title(f"y={int(ys[i])}, ŷ={int(preds[i])}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        return fig

    @output
    @render.text
    def metrics_text():
        m = trained_model()
        b = data_bundle()
        if m is None or b["kind"].startswith("toy") or b["kind"] == "blob2d":
            return "Metrics: (image classification only)"
        te = b["test"]
        device = next(m.parameters()).device
        m.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for X,y in te:
                logits = m(X.to(device))
                pred = logits.argmax(dim=1).cpu()
                correct += int((pred==y).sum())
                total   += len(y)
        acc = correct/max(1,total)
        return f"Test accuracy: {acc:.3f} ({correct}/{total})"

    @output
    @render.plot
    def conf_mat_plot():
        m = trained_model()
        b = data_bundle()
        fig, ax = plt.subplots()
        cmap = plt.get_cmap("viridis")

        if m is None or b["kind"].startswith("toy") or b["kind"].startswith("blob2d"):
            ax.set_title("Train a model first")
            ax.grid(False)
            return fig
        
        # Build confusion matrix:
        te = b["test"]
        device = next(m.parameters()).device
        m.eval()
        # 10-class assumption for MNIST/FashionMNIST
        K = 10
        C = np.zeros((K,K), dtype=int)
        with torch.no_grad():
            for X,y in te:
                logits = m(X.to(device))
                pred = logits.argmax(dim=1).cpu().numpy()
                yy   = y.numpy()
                for p,t in zip(pred, yy): C[t,p] += 1
        if bool(input.cm_norm()):
            Cn = C / np.maximum(1, C.sum(axis=1, keepdims=True))
            im = ax.imshow(Cn, vmin=0, vmax=1, cmap=cmap)
            ax.set_title("Confusion matrix (normalized)")
        else:
            im = ax.imshow(C, cmap=cmap)
            ax.set_title("Confusion matrix")

        ax.grid(False)

        # show the numbers
        for i in range(K):
            for j in range(K):
                if bool(input.cm_norm()):
                    ax.text(j, i, Cn[i,j].round(2), ha="center", va="center", color="white" if i != j else "black")
                else:
                    ax.text(j, i, C[i,j], ha="center", va="center", color="white" if i != j else "black")
        # no axis ticks
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        return fig

    @output
    @render.plot
    def input_weights_plot():
        b = data_bundle()
        fig = plt.figure()
        m = trained_model()

        # Guards
        if m is None:
            plt.title("Train a model first")
            plt.grid(False)
            return fig
        if b.get("kind", "").startswith("toy") or b.get("kind", "").startswith("blob2d"):
            plt.title("Only available for image models (MNIST/FashionMNIST)")
            plt.grid(False)
            return fig

        try:
            x0, _ = next(iter(b["train"]))
            # x0 shapes typically: (N, C, H, W) or (N, 1, H, W)
            if x0.ndim == 4:
                _, C, H, W = x0.shape
            elif x0.ndim == 3:
                # (N, H, W) rare; assume single channel
                _, H, W = x0.shape
                C = 1
            else:
                raise ValueError(f"Unexpected input tensor shape: {tuple(x0.shape)}")
        except Exception:
            # Fallback for MNIST-like shapes
            C, H, W = 1, 28, 28

        target_linear = None
        HW = H * W
        for layer in m.modules():
            if isinstance(layer, nn.Linear) and getattr(layer, "in_features", None) == HW:
                target_linear = layer
                break
        if target_linear is None:
            # Fallback: first Linear in the model
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    target_linear = layer
                    break

        if target_linear is None:
            plt.title("No nn.Linear layer found in model")
            return fig

        try:
            u6.plot_input_weights(target_linear, input_shape=(H, W), max_num=25, ncols=5, image_size=2.0)
        except Exception as e:
            plt.title(f"Input weight visualization error:\n{e}")
        return plt.gcf()

    @render.text
    def param_values():
        m = trained_model()
        if m is None:
            return "Parameter values: (train a model first)"

        # total parameter count
        n_params = sum(p.numel() for p in m.parameters())
        if n_params > 10:
            return f"Model has {n_params} parameters → not displaying (limit: 10)."

        # Collect flattened parameter values + shapes
        lines = [f"Total parameters: {n_params}", ""]
        sd = m.state_dict()
        for name, t in sd.items():
            arr = t.detach().cpu().numpy()
            flat = arr.reshape(-1)
            # keep it readable
            vals = ", ".join(f"{float(v): .6g}" for v in flat.tolist())
            lines.append(f"{name:30s} shape={tuple(arr.shape)}  values=[{vals}]")

        # Extra: interpret a simple 1D regression head (Linear: in=1, out=1)
        # For a single linear neuron: y = d + k*x
        try:
            w = None
            b = None
            for layer in m.modules():
                if isinstance(layer, nn.Linear) and layer.in_features == 1 and layer.out_features == 1:
                    w = layer.weight.detach().cpu().view(-1).numpy()
                    b = layer.bias.detach().cpu().view(-1).numpy() if layer.bias is not None else None
                    break
            if w is not None:
                k = float(w[0])
                d = float(b[0]) if b is not None else 0.0
                lines += ["", f"Interpreting as 1D linear model:  y = d + k·x",
                          f"  k (slope)     = {k:.6g}",
                          f"  d (intercept) = {d:.6g}"]
        except Exception:
            pass

        return "\n".join(lines)



app = App(app_ui, server)

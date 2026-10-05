from pathlib import Path
from shiny import App, ui, render, reactive
from shinywidgets import output_widget, render_widget, render_plotly
import plotly.express as px
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import spacy
import pandas as pd
import u2_utils as u2
import math

import io, base64
from scipy.io import wavfile
import plotly.graph_objects as go
from scipy.signal import stft
import subprocess
import sys
import pkg_resources
from sklearn.decomposition import PCA

from scipy.signal import resample_poly

import numpy as np


def _np_to_wav_bytes(points: np.ndarray, sr: int) -> bytes:
    """Converts float [-1,1] or any scaled to 16-bit PCM WAV bytes."""
    # Normalize and convert to int16
    if points.dtype != np.float32 and points.dtype != np.float64:
        points = points.astype(np.float32)
    maxabs = np.max(np.abs(points)) or 1.0
    y = np.clip(points / maxabs, -1.0, 1.0)
    y16 = (y * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    wavfile.write(buf, sr, y16)
    return buf.getvalue()



def project_embeddings(emb: np.ndarray, labels=None, texts=None, n_components=2):
    """
    emb: (N, D) numpy array
    returns DataFrame with columns ['x','y'] or ['x','y','z'] plus optional 'label','text'
    """
    emb = np.asarray(emb, dtype=float)
    k = 3 if n_components == 3 else 2
    coords = PCA(n_components=k, random_state=0).fit_transform(emb)

    df = pd.DataFrame(coords, columns=["x", "y"] + (["z"] if k == 3 else []))
    if labels is not None:
        df["label"] = labels
    df["text"] = texts if texts is not None else np.arange(len(df)).astype(str)
    return df


def install_spacy_model(model_name="en_core_web_md"):
    try:
        pkg_resources.get_distribution(model_name)
    except pkg_resources.DistributionNotFound:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])

install_spacy_model("en_core_web_md")
nlp = spacy.load("en_core_web_md")



app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.script(src="vendor/plotly/plotly.min.js"),
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
        ui.tags.script("""
            document.addEventListener('DOMContentLoaded', function(){
            function wirePlotClick(){
                const player = document.getElementById('player');
                const gd = document.querySelector('[data-output-id="sig_plot_player"] .js-plotly-plot');
                if(!player || !gd){ setTimeout(wirePlotClick, 300); return; }
                gd.on('plotly_click', function(eventData){
                if (eventData && eventData.points && eventData.points[0]) {
                    const x = eventData.points[0].x; // Sekunden
                    player.currentTime = x;
                }
                });
            }
            wirePlotClick();
            });
        """),
        ui.tags.script("""
            document.addEventListener("shown.bs.tab", function(){
            const t = Date.now();
            // fire immediately
            Shiny.setInputValue("sig_tab_shown", t, {priority:"event"});
            // fire again shortly after binding finishes
            setTimeout(() => {
                Shiny.setInputValue("sig_tab_shown", t + 1, {priority:"event"});
            }, 120);
            }, true);
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
    ui.h2("Hands-on AI I - Unit 2: Data Types", class_="text-center mb-4"),
    ui.navset_tab(
        ui.nav_panel("Image Processing",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_file("img_file", "Upload Image", accept=[".jpg", ".jpeg", ".png"]),
                    ui.input_select("img_op", "Operation", {
                        "show": "Show Image",
                        "channels": "Show RGB Channels",
                        "rgba": "Use RGBA",
                        "grayscale": "Show Grayscale",
                        "rotate": "Rotate",
                        "flip": "Flip",
                        "crop": "Crop",
                        "blur": "Blur",
                        "hist": "Color Histograms",
                        "segment": "Segment"
                    }),
                    ui.panel_conditional("input.img_op == 'show'",
                        ui.output_text("image_dim"),
                        ui.input_select("channel", "Channel", {"R": "Red", "G": "Green", "B": "Blue"}, selected="R"),
                        ui.output_text("image_max_min")
                    ),

                    ui.panel_conditional("input.img_op == 'rotate'",
                        ui.input_numeric("rotate_angle", "Angle (deg)", value=45)
                    ),
                    ui.panel_conditional("input.img_op == 'rgba'",
                        ui.input_slider("alpha", "Alpha (0-1)", 0, 1, 1, step=0.01)
                    ),

                    ui.panel_conditional("input.img_op == 'flip'",
                        ui.input_select("flip_dir", "Direction", {"horizontal": "Horizontal", "vertical": "Vertical"})
                    ),
                    ui.panel_conditional("input.img_op == 'crop'",
                        ui.input_numeric("crop_left", "Left", value=0),
                        ui.input_numeric("crop_top", "Top", value=0),
                        ui.input_numeric("crop_width", "Width", value=100),
                        ui.input_numeric("crop_height", "Height", value=100)
                    ),
                    ui.panel_conditional("input.img_op == 'blur'",
                        ui.input_slider("blur_sigma", "Sigma (Blur)", 0, 10, 1, step=0.1)
                    ),
                    ui.panel_conditional("input.img_op == 'segment'",
                        ui.input_slider("seg_r", "Red threshold", 0, 255, value=[0, 255]),
                        ui.input_slider("seg_g", "Green threshold", 0, 255, value=[0, 255]),
                        ui.input_slider("seg_b", "Blue threshold", 0, 255, value=[0, 255])
                    ),
                ),
                ui.output_plot("img_plot", height=600)
            )
        ),
        ui.nav_panel("Signal Processing",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_select("sig_op", "Operation", {
                        "generate": "Generate Sine Wave - Single",
                        "generate_multi": "Generate Sine Wave - Multi",
                        "upload": "Upload WAV File"
                    }, selected="generate"),
                    ui.input_select("sig_plot", "Plot Type", {
                        "wave": "Waveform",
                        "spectrum": "Fourier Spectrum",
                        "spectrogram": "Spectrogram"
                    }, selected="wave"),

                    ui.panel_conditional("input.sig_op == 'generate'",
                        ui.input_numeric("sig_freq", "Frequency (Hz)", value=1)
                    ),
                    ui.panel_conditional("input.sig_op == 'generate_multi'",
                        ui.input_text("sig_freq_multi", "Frequencies (Hz, comma-separated)", value="440,349.228,261.626")
                    ),
                    ui.panel_conditional("input.sig_op == 'generate' || input.sig_op == 'generate_multi'",
                        ui.input_numeric("sig_time", "Duration (s)", value=1),
                        ui.input_numeric("sig_sr", "Sampling Rate", value=24000)
                    ),
                    ui.panel_conditional("input.sig_op == 'generate'",
                        ui.input_checkbox("show_sampling_points", "Show Sampling Points", value=False),
                    ),
                    ui.panel_conditional("input.sig_op == 'upload'",
                        ui.input_file("wav_file", "Upload WAV", accept=[".wav"])
                    ),                   
                    ui.input_numeric("max_freq", "Max Frequency (for spectrum)", value=2000),
                    ui.input_select("view_mode", "View", {
                        "static": "Static (matplotlib)",
                        "player": "Interactive (plotly)"
                    }, selected="player"),     
                ),
                ui.panel_conditional("input.view_mode == 'player'",
                    ui.card(
                        ui.card_header("Audio Player"),
                        ui.output_ui("audio_player")
                    )
                ),
                ui.panel_conditional("input.view_mode == 'player' && input.sig_plot == 'wave'",             
                    ui.card(ui.card_header("Waveform"), ui.output_ui("sig_plot_player")),   
                ),
                ui.panel_conditional("input.view_mode == 'player' && input.sig_plot == 'spectrum'",
                    ui.card(ui.card_header("Spectrum"), ui.output_ui("spectrum_plot"))
                ),
                ui.panel_conditional("input.view_mode == 'player' && input.sig_plot == 'spectrogram'",
                    ui.card(ui.card_header("Spectrogram"), ui.output_ui("spectrogram_plot"))
                ),
                ui.br(),
                ui.panel_conditional("input.view_mode == 'static'",
                    ui.panel_conditional("input.sig_plot == 'wave'",
                        ui.card(ui.card_header("Waveform"), ui.output_plot("waveform_plot_static", height=600)),
                    ),
                    ui.panel_conditional("input.sig_plot == 'spectrum'",
                        ui.card(ui.card_header("Spectrum"), ui.output_plot("spectrum_plot_static", height=600)),
                    ),
                    ui.panel_conditional("input.sig_plot == 'spectrogram'",
                        ui.card(ui.card_header("Spectrogram"), ui.output_plot("spectrogram_plot_static", height=600)),
                    )
                )
            ),
        ),
        ui.nav_panel("Text Processing",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_text_area("word_list", "Vocabulary (comma-separated)", "dog,cat,tiger,lion,car,bike,apple,banana,jeans,dress,man,woman,king,queen", rows=5),
                    ui.input_action_button("run_embed", "Load vocabulary", class_="btn-primary"),
                ),
                ui.navset_pill(
                    ui.nav_panel("One-Hot Encoding",
                        ui.p(),
                        ui.panel_conditional("input.run_embed",
                            ui.h4("Vocabulary Sequence:"),
                            ui.output_text("vocab_sequence"),
                            ui.p(),
                            ui.h4("One-Hot Encoding Table:"),
                            ui.output_data_frame("one_hot_table"),
                            ui.p(),
                            ui.p("Now we can create a one-hot encoded representation of a phrase:"),
                            ui.input_text("phrase", "Enter a phrase (e.g. 'Paul likes to play tennis')", placeholder="Paul likes to play tennis", width="400px"),
                            ui.output_data_frame("one_hot_phrase_table"),
                            ui.p(),
                            ui.input_action_button("run_one_hot_phrase", "Run One-Hot Encoding for Phrase", class_="btn-primary")
                        ),
                    ),
                    ui.nav_panel("Word Embeddings",
                        ui.p(),
                        ui.panel_conditional("input.run_embed",
                            ui.h4("Word Embeddings DataFrame:"),
                            ui.output_data_frame("embed_df")
                        ),
                    ),
                    ui.nav_panel("Plot Embeddings",
                        ui.p(),
                        ui.panel_conditional("input.run_embed",
                            ui.layout_columns(
                                ui.input_select(
                                    "emb_view", "Embedding view",
                                    {"2d": "2D", "3d": "3D"},
                                    selected="2d"
                                ),
                                ui.input_action_button("embed_plot_btn", "Create/Update Plot", class_="btn-primary"),
                                col_widths=(4, 2),
                                class_="d-flex align-items-end"
                            ),
                            ui.panel_conditional("input.emb_view == '2d'",
                                ui.card(ui.card_header("Embeddings (2D)"),
                                        ui.output_ui("embedding_plot_2d"))
                            ),
                            ui.panel_conditional("input.emb_view == '3d'",
                                ui.card(ui.card_header("Embeddings (3D)"),
                                        ui.output_ui("embedding_plot_3d"))
                            ),
                        ),
                    ),
                    ui.nav_panel("Individual Words",
                        ui.p(),
                        ui.panel_conditional("input.run_embed",
                            ui.input_select("word_select", label="Select a word", choices=[], selected=None),
                            ui.panel_conditional("input.word_select",
                                ui.output_code("word_vector_size"),
                                ui.output_code("word_vector")
                            ),
                        )
                    ),
                    ui.nav_panel("Word Similarity",
                        ui.p(),
                        ui.panel_conditional("input.run_embed",
                            ui.h4("Compare Word Similarity"),
                        ui.p(),
                        ui.input_select("word_select1", "Select word 1", choices=[], selected=None),
                        ui.output_text("similarity_text"),
                        ui.p(),
                        ui.input_select("word_select2", "Select word 2", choices=[], selected=None),
                        ui.panel_conditional("input.find_similarity",
                            ui.output_code("word_similarity")
                        ),
                        ui.input_action_button("find_similarity", "Find Similarity", class_="btn-primary"),

                        ui.p(),
                        ui.h4("Find Similar Words"),
                        ui.input_select("word_select_3", "Select a word", choices=[], selected=None),
                        ui.panel_conditional("input.find_similar",
                            ui.output_code("similar_words")
                        ),
                        ui.p(),
                        ui.input_action_button("find_similar", "Find Similar Words", class_="btn-primary"),

                        ui.p(),
                        ui.h4("Compute Word"),
                        ui.input_text("word_computation", "Enter a computation (e.g 'king - man + woman')",
                                      placeholder="king - man + woman", width="400px"),
                        ui.panel_conditional("input.compute_word",
                            ui.output_code("computed_words")
                        ),
                        ui.p(),
                        ui.input_action_button("compute_word", "Compute Word", class_="btn-primary")
                        )
                    )
                )
            )
        )
    )
)


def get_uploaded_image_path(file_input):
    if not file_input:
        return None
    fileinfo = file_input[0]
    return fileinfo["datapath"]

def get_uploaded_wav_path(file_input):
    if not file_input:
        return None
    fileinfo = file_input[0]
    return fileinfo["datapath"]


def server(input, output, session):

    @output
    @render.plot
    def img_plot():
        img_path = get_uploaded_image_path(input.img_file())
        if not img_path:
            plt.text(0.5, 0.5, "Upload an image", ha="center")
            plt.axis('off')
            return plt.gcf()
        op = input.img_op()
        if op == "show":
            u2.plot_image(img_path)
        elif op == "channels":
            u2.plot_image_channels_rgb(img_path)
        elif op == "rgba":
            u2.plot_image_rgba(img_path, alpha=input.alpha())
        elif op == "grayscale":
            u2.plot_image_grayscale(img_path)
        elif op == "rotate":
            u2.plot_rotated_image(img_path, angle=input.rotate_angle())
        elif op == "flip":
            u2.plot_flipped_image(img_path, flipping=input.flip_dir())
        elif op == "crop":
            u2.plot_cropped_image(img_path, left=input.crop_left(), top=input.crop_top(),
                                  width=input.crop_width(), height=input.crop_height())
        elif op == "blur":
            u2.plot_blurred_image(img_path, sigma=input.blur_sigma())
        elif op == "hist":
            u2.plot_color_histograms(img_path)
        elif op == "segment":
            u2.segment_image(img_path, upper_threshold_r=input.seg_r()[1], upper_threshold_g=input.seg_g()[1], upper_threshold_b=input.seg_b()[1], lower_threshold_b=input.seg_b()[0], lower_threshold_g=input.seg_g()[0], lower_threshold_r=input.seg_r()[0])
        else:
            plt.text(0.5, 0.5, "Unknown operation", ha="center")
        return plt.gcf()
    
    @output
    @render.text
    def image_dim():
        img_path = get_uploaded_image_path(input.img_file())
        if not img_path:
            return "No image uploaded"
        dimensions = u2.get_image_dimensions(img_path)
        return f"({dimensions[0]}, {dimensions[1]}, {dimensions[2]})"
    
    @output
    @render.text
    def image_max_min():
        img_path = get_uploaded_image_path(input.img_file())
        if not img_path:
            return "No image uploaded"
        max_min = u2.get_image_max_min(img_path, input.channel())
        return f"Max: {max_min[0]}, Min: {max_min[1]}"


    def _val(v, default):
        return default if v in (None, "", []) else v

    @reactive.calc
    def get_signal():
        op = input.sig_op() or "generate"

        if op == "generate":
            sr = int(_val(input.sig_sr(), 24000))
            duration = float(_val(input.sig_time(), 1))
            freq = float(_val(input.sig_freq(), 1))
            points = u2.generate_wave(freq, duration, sr)
            return points, sr, duration

        elif op == "generate_multi":
            sr = int(_val(input.sig_sr(), 24000))
            duration = float(_val(input.sig_time(), 1))
            freqs_str = _val(input.sig_freq_multi(), "440,349.228,261.626")
            freqs = [float(f) for f in freqs_str.split(",") if f.strip()]
            waves = [u2.generate_wave(f, duration, sr) for f in freqs]
            points = np.sum(waves, axis=0)
            return points, sr, duration

        elif op == "upload" and input.wav_file():
            wav_path = get_uploaded_wav_path(input.wav_file())
            points, sr = u2.read_wav_file(wav_path)
            return points, sr, len(points) / sr

        return None, None, None


    @output
    @render.plot
    def waveform_plot_static():
        points, sr, duration = get_signal()
        if points is None:
            plt.text(0.5, 0.5, "Generate or upload a signal", ha="center")
            return plt.gcf()

        # 1) Smooth, band-limited line (works for generated & uploaded)
        u2.plot_wave(points, duration, sr)

        # 2) Optional sampling points overlay
        if input.sig_op() == "generate" and input.show_sampling_points():
            n = len(points)
            t_samples = np.linspace(0, duration, n, endpoint=False)
            # For uploaded/long signals, don’t flood the canvas:
            step = 1 if input.sig_op() in ("generate", "generate_multi") else max(1, n // 1000)
            plt.plot(t_samples[::step], points[::step], "ro", markersize=3, label="Sampling points")
            plt.legend()

        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title(f"Waveform (Sampling Rate: {sr} Hz), Wave Points: {len(points)}")
        return plt.gcf()


    @output
    @render.plot
    def spectrogram_plot_static():
        points, sr, duration = get_signal()
        if points is None:
            plt.text(0.5, 0.5, "Generate or upload a signal", ha="center")
            return plt.gcf()

        spec = u2.compute_spectrogram(points)
        u2.plot_spectrogram(spec, sr, max_freq=float(input.max_freq()))

        return plt.gcf()
    
    @output
    @render.plot
    def spectrum_plot_static():
        points, sr, duration = get_signal()
        if points is None:
            plt.text(0.5, 0.5, "Generate or upload a signal", ha="center")
            return plt.gcf()

        ft = u2.apply_fourier_transform(points)
        u2.plot_spectrum(ft, sr, max_freq=float(input.max_freq()))

        return plt.gcf()
    
    @output
    @render.ui
    def audio_player():
        points, sr, duration = get_signal()
        if points is None or sr is None:
            return ui.HTML("<em>Generate or upload a signal to enable the player.</em>")

        if input.sig_op() in ("generate", "generate_multi"):
            wav_bytes = _np_to_wav_bytes(points, sr)
        else:
            # upload: use original file bytes
            wav_path = get_uploaded_wav_path(input.wav_file())
            with open(wav_path, "rb") as f:
                wav_bytes = f.read()

        b64 = base64.b64encode(wav_bytes).decode("ascii")
        # HTML5-Player + JS to report playhead position
        return ui.HTML(f"""
        <audio id="player" controls style="width:100%">
        <source src="data:audio/wav;base64,{b64}" type="audio/wav" />
        Your browser does not support the audio element.
        </audio>
        <script>
        (function(){{
        function hook(){{
            const a = document.getElementById('player');
            if(!a) {{ setTimeout(hook, 200); return; }}
            setInterval(() => {{
            if (!isNaN(a.currentTime)) {{
                Shiny.setInputValue('playhead_sec', a.currentTime, {{priority:'event'}});
                Shiny.setInputValue('audio_duration', a.duration || null, {{priority:'event'}});
                Shiny.setInputValue('audio_paused', a.paused, {{priority:'event'}});
            }}
            }}, 100);
        }}
        if (document.readyState !== 'loading') hook();
        else document.addEventListener('DOMContentLoaded', hook);
        }})();
        </script>
        """)
    
    _playhead = reactive.Value(0.0)

    def _downsample(t: np.ndarray, y: np.ndarray, target=8000):
        if len(y) <= target:
            return t, y
        step = int(np.ceil(len(y)/target))
        return t[::step], y[::step]

    @reactive.effect
    @reactive.event(input.playhead_sec, ignore_init=True)
    def _update_playhead():
        try:
            _playhead.set(float(input.playhead_sec()))
        except Exception:
            pass

    @reactive.effect
    @reactive.event(input.sig_op)
    def _reset_sampling_points_if_not_single():
        if input.sig_op() != "generate":
            try:
                ui.update_checkbox("show_sampling_points", value=False)
            except Exception:
                pass

    @output
    @render.ui
    def sig_plot_player():
        _ = (input.view_mode(), input.sig_plot(), input.sig_tab_shown(), input.sig_op(), input.show_sampling_points())

        if input.view_mode() != "player" or input.sig_plot() != "wave":
            return ui.HTML("<div></div>")

        points, sr, duration = get_signal()
        if points is None or sr is None:
            return ui.HTML("<em>Generate or upload a signal to enable the player.</em>")

        target = 8000
        n = len(points)
        step = int(np.ceil(n / target)) if n > target else 1
        y_ds = points[::step]
        t_ds = np.arange(len(y_ds)) * step / sr


        fig = go.Figure()

        n = points.size
        max_line_pts = 10000
        up = min(16, max(1, math.ceil(max_line_pts / max(1, n))))

        # Band-limited interpolation via polyphase resampling
        y_smooth = resample_poly(points, up, 1)             # smooth continuous-time reconstruction
        t_smooth = np.arange(y_smooth.size) / (sr * up)     # in seconds

        fig.add_trace(
            go.Scatter(
                x=t_smooth,
                y=y_smooth,
                mode="lines",
                name="Waveform (band-limited)",
                hoverinfo="skip",
            )
        )

        if input.sig_op() == "generate" and input.show_sampling_points():

            t_all = np.arange(n) / sr
            max_markers = 100_000
            step_mark = max(1, math.ceil(n / max_markers))
            fig.add_trace(
                go.Scattergl(
                    x=t_all[::step_mark],
                    y=points[::step_mark],
                    mode="markers",
                    name="Sampling points",
                    marker=dict(size=3),
                    hovertemplate="t=%{x:.6f}s<br>y=%{y:.6g}<extra>sample</extra>",
                )
            )


        

        y_min = float(np.min(y_ds)) if len(y_ds) else -1.0
        y_max = float(np.max(y_ds)) if len(y_ds) else 1.0
        if y_min == y_max:
            y_min, y_max = -1.0, 1.0

        # Playhead line
        fig.add_shape(
            type="line",
            x0=0, x1=0,
            y0=y_min, y1=y_max,
            line=dict(color="red", width=2)
        )
        fig.update_layout(
            showlegend=False,
            margin=dict(l=30, r=20, t=30, b=30),
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            clickmode="event+select",
            dragmode="zoom",
            height=420,
        )

        plot_html = fig.to_html(full_html=False, include_plotlyjs=False, div_id="wave_plot_div")

        js = """
        <script>
        (function () {
        const DIV_ID    = "wave_plot_div";
        const TIMER_KEY = "__playheadTimer";
        function getDiv()    { return document.getElementById(DIV_ID); }
        function getPlayer() { return document.getElementById("player"); }

        function startTimer() {
            const player = getPlayer();
            if (!player) return;

            if (isNaN(player.duration) || player.duration === Infinity) {
            player.addEventListener("loadedmetadata", startTimer, { once: true });
            return;
            }
            if (window[TIMER_KEY]) { clearInterval(window[TIMER_KEY]); window[TIMER_KEY] = null; }

            window[TIMER_KEY] = setInterval(() => {
            const div = getDiv();
            if (!div || !window.Plotly || isNaN(player.currentTime)) return;
            const shapes = (div.layout && div.layout.shapes) || [];
            if (!shapes || !shapes.length) return;
            const t = player.currentTime;
            window.Plotly.relayout(div, { "shapes[0].x0": t, "shapes[0].x1": t });
            }, 50);
        }

        function wireClickToSeek() {
            const div = getDiv();
            const player = getPlayer();
            if (!div || !player || !window.Plotly) return;
            if (div.__clickBound) return;
            div.__clickBound = true;
            div.on?.("plotly_click", ev => {
            const x = ev?.points?.[0]?.x;
            if (!isNaN(x)) player.currentTime = x;
            });
        }

        function bootWhenVisible() {
            const div = getDiv();
            if (!div || !window.Plotly) { setTimeout(bootWhenVisible, 100); return; }
            const tryStart = () => {
            const visible = div.offsetParent !== null && div.clientWidth > 0;
            if (!visible) { setTimeout(tryStart, 80); return; }
            try { if (div.isConnected && div.offsetParent !== null && div.clientWidth > 0) window.Plotly.Plots.resize(div).catch(() => undefined); } catch (e) {}
            startTimer();
            wireClickToSeek();
            if (!div.__resizeBound) {
                div.__resizeBound = true;
                new ResizeObserver(() => { try { if (div.isConnected && div.offsetParent !== null && div.clientWidth > 0) window.Plotly.Plots.resize(div).catch(() => undefined); } catch (e) {} }).observe(div);
            }
            };
            tryStart();
        }

        // Kick things off after this HTML is mounted
        if (document.readyState !== "loading") bootWhenVisible();
        else document.addEventListener("DOMContentLoaded", bootWhenVisible);

        // Re-run when Bootstrap tabs show
        document.addEventListener("shown.bs.tab", () => setTimeout(bootWhenVisible, 50), true);

        // Restart timer when audio source changes
        const p = getPlayer();
        if (p && !p.__restartBound) {
            p.__restartBound = true;
            ["loadedmetadata","emptied"].forEach(evt => p.addEventListener(evt, startTimer));
        }
        })();
        </script>
        """
        return ui.HTML(plot_html + js)
    


    @output
    @render.ui
    def spectrum_plot():
        _ = input.sig_tab_shown()
        if input.view_mode() != "player":
            return ui.HTML("")
        points, sr, duration = get_signal()
        if points is None:
            return ui.HTML("<em>Generate or upload a signal first.</em>")

        freqs_hz, mag = compute_spectrum_linear(points, sr)
        return plotly_spectrum_html_linear(freqs_hz, mag, max_freq=input.max_freq() or None)

    
    def compute_spectrum_linear(points: np.ndarray, sr: int):
        """
        Match Matplotlib spectrum: single-sided linear amplitude vs frequency (Hz).
        """
        y = np.asarray(points, dtype=float)
        N = len(y)
        Y = np.fft.rfft(y)
        mag = np.abs(Y) / N              # linear amplitude (no dB)
        freqs_hz = np.fft.rfftfreq(N, d=1.0/sr)  # Hz
        return freqs_hz, mag

    def plotly_spectrum_html_linear(freqs_hz, mag, max_freq=None, div_id="spectrum_div"):
        # Crop to max_freq like Matplotlib
        if max_freq:
            idx = np.searchsorted(freqs_hz, float(max_freq))
            freqs_hz = freqs_hz[:idx]
            mag = mag[:idx]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=freqs_hz, y=mag, mode="lines", name="Spectrum"))
        fig.update_layout(
            xaxis_title="freq (Hz)",
            yaxis_title="amplitude",
            margin=dict(l=50, r=20, t=30, b=50),
            showlegend=False,
            height=420,
        )
        return ui.HTML(fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id))


    def compute_spectrogram_log(points: np.ndarray, sr: int, winsize=1024, hop=512):
        """
        Match Matplotlib spectrogram: log(|STFT|), gray colormap, origin='lower', x = time frames.
        Returns (freqs_hz, frame_indices, S_log, scaler), where scaler = sr/hop for sec to frame mapping.
        """
        f, t, Zxx = stft(points, fs=sr, nperseg=winsize, noverlap=winsize-hop,
                        window="hann", boundary=None, padded=False)
        S = np.abs(Zxx)
        # Emulate LogNorm lower floor ~ 1e-3 to log10 range [-3, 0]
        S = np.maximum(S, 1e-3)
        S_log = np.log10(S)  # z in [-3, 0]
        frames = np.arange(S_log.shape[1])  # frame indices instead of seconds
        scaler = sr / hop                    # seconds to frames : round(sec * scaler)
        return f, frames, S_log, scaler

    def plotly_spectrogram_html_log(freqs_hz, frames, S_log, max_freq=None, scaler=None, div_id="specgram_div"):
        # Optional frequency crop like Matplotlib
        if max_freq:
            idx = np.searchsorted(freqs_hz, float(max_freq))
            freqs_hz = freqs_hz[:idx]
            S_log = S_log[:idx, :]

        heat = go.Heatmap(
            z=S_log,
            x=frames,  # frames (integer index)
            y=freqs_hz,  # Hz
            zmin=-2, zmax=0,  # ~log10(1e-2) .. log10(1)
            colorscale=[[0.0, "black"], [1.0, "white"]],  # grayscale like plt.cm.gray (white=high)
            colorbar=dict(title="log mag")
        )
        fig = go.Figure(data=heat)
        fig.update_layout(
            xaxis_title="time frame",
            yaxis_title="freq (Hz)",
            margin=dict(l=60, r=20, t=30, b=50),
            height=500
        )
        # Playhead at frame 0 (moved via JS)
        fig.add_shape(type="line", x0=0, x1=0, y0=float(freqs_hz.min()), y1=float(freqs_hz.max()),
                    line=dict(color="red", width=2))

        html = fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)


        move_js = ""
        if scaler is not None:
            move_js = f"""
            <script>
            (function(){{
            const DIV_ID="{div_id}";
            const TIMER_KEY="__playheadTimer_spec";
            const SCALE={scaler:.6f}; // frames per second
            const getDiv=()=>document.getElementById(DIV_ID);
            const getPlayer=()=>document.getElementById("player");

            function startTimer(){{
                const p=getPlayer();
                if(!p) return;
                if(isNaN(p.duration) || p.duration===Infinity){{
                p.addEventListener("loadedmetadata", ()=>startTimer(), {{once:true}});
                return;
                }}
                if(window[TIMER_KEY]){{ clearInterval(window[TIMER_KEY]); window[TIMER_KEY]=null; }}
                window[TIMER_KEY]=setInterval(()=>{{
                const d=getDiv(); if(!d||!window.Plotly||isNaN(p.currentTime)) return;
                const shapes=(d.layout&&d.layout.shapes)||[]; if(!shapes.length) return;
                const frame=Math.round(p.currentTime * SCALE);
                window.Plotly.relayout(d, {{"shapes[0].x0": frame, "shapes[0].x1": frame}});
                }}, 50);
            }}

            function bootWhenVisible(){{
                const d=getDiv(); if(!d||!window.Plotly){{ setTimeout(bootWhenVisible,100); return; }}
                const tryStart=()=>{{
                const visible=d.offsetParent!==null && d.clientWidth>0;
                if(!visible){{ setTimeout(tryStart,80); return; }}
                try{{if (d.isConnected && d.offsetParent !== null && d.clientWidth > 0) window.Plotly.Plots.resize(d).catch(() => undefined);}}catch(e){{}}
                startTimer();
                if(!d.__resizeBound){{
                    d.__resizeBound=true;
                    new ResizeObserver(()=>{{ try{{if (d.isConnected && d.offsetParent !== null && d.clientWidth > 0) window.Plotly.Plots.resize(d).catch(() => undefined);}}catch(e){{}} }}).observe(d);
                }}
                }};
                tryStart();
            }}

            if(document.readyState!=="loading") bootWhenVisible();
            else document.addEventListener("DOMContentLoaded", bootWhenVisible);
            document.addEventListener("shown.bs.tab", ()=>setTimeout(bootWhenVisible,50), true);
            const p=getPlayer();
            if(p && !p.__restartBound_spec2){{
                p.__restartBound_spec2=true;
                ["loadedmetadata","emptied"].forEach(evt=>p.addEventListener(evt, ()=>startTimer()));
            }}
            }})();
            </script>
            """
        return ui.HTML(html + move_js)


    @output
    @render.ui
    def spectrogram_plot():
        _ = input.sig_tab_shown()
        if input.view_mode() != "player":
            return ui.HTML("")
        points, sr, duration = get_signal()
        if points is None:
            return ui.HTML("<em>Generate or upload a signal first.</em>")
        f_hz, frames, S_log, scaler = compute_spectrogram_log(points, sr, winsize=1024, hop=512)
        return plotly_spectrogram_html_log(
            f_hz, frames, S_log,
            max_freq=(input.max_freq() or None),
            scaler=scaler
        )


    # Word Embeddings
    @reactive.calc
    def get_words():
        words = [w.strip() for w in input.word_list().split(",") if w.strip()]
        # update word choices for the word_select input
        ui.update_select("word_select", choices=words, selected=None)
        ui.update_select("word_select2", choices=words, selected=None)
        ui.update_select("word_select1", choices=words, selected=None)
        ui.update_select("word_select_3", choices=words, selected=None)
        return words

    @reactive.calc
    def get_word_vecs():
        words = get_words()
        if not words:
            return pd.DataFrame()
        vecs = u2.get_word_vectors(nlp, words)
        print(vecs)
        return vecs
    
    @output
    @render.code
    @reactive.event(input.find_similar)
    def similar_words():
        word = input.word_select_3()
        if not word:
            return "Select a word to find similar words."
        
        word_vec = nlp(word).vector
        if word_vec is None or len(word_vec) == 0:
            return f"Word '{word}' has no vector representation."
        df = get_word_vecs()
        print(df)
        print(word_vec)
        similar, distant = u2.find_similar_words(word_vec, get_word_vecs())
        return f"Similar words to '{word}': {', '.join(similar)}\nDistant words: {', '.join(distant)}"

    @output
    @render.code
    @reactive.event(input.compute_word)
    def computed_words():
        computation = input.word_computation()
        parts = computation.split()
        embeddings = []
        for part in parts:
            if part in ["+", "-", "*", "/"]:
                embeddings.append(part)
            else:
                word_vec = nlp(part).vector
                if word_vec is None or len(word_vec) == 0:
                    return f"Word '{part}' has no vector representation."
                embeddings.append(word_vec)
        if len(embeddings) < 3:
            return "Enter a valid computation with at least two words and one operator."
        result = embeddings[0]
        for i in range(1, len(embeddings)):
            if isinstance(embeddings[i], str):
                if embeddings[i] == "+":
                    result += embeddings[i + 1]
                elif embeddings[i] == "-":
                    result -= embeddings[i + 1]
                else:
                    return f"Unknown operator '{embeddings[i]}' in computation."
        if result is None or len(result) == 0:
            return "Computation resulted in an empty vector."
        similar, _ = u2.find_similar_words(result, get_word_vecs())
        return f"Computed words for '{computation}': {', '.join(similar)}"

    @output
    @render.text
    def similarity_text():
        return f"Choose a word to compare {input.word_select1()} to:"


    @output
    @render.code
    @reactive.event(input.find_similarity)
    def word_similarity():
        word1 = input.word_select1()
        word2 = input.word_select2()
        if not word1 or not word2:
            return "Select two words to compare."
        
        word1 = nlp(word1)
        word2 = nlp(word2)
        
        if word1.vector_norm == 0 or word2.vector_norm == 0:
            return "One or both words have no vector representation."
        sim = word1.similarity(word2)
        if np.isnan(sim):
            return "Similarity could not be computed (NaN)."

        return f"Similarity between '{word1}' and '{word2}': {sim:.3f}"

    @output
    @render.code
    def word_vector_size():
        word = input.word_select()
        if not word:
            return "Select a word to see its vector size."
        
        word_vec = nlp(word).vector
        if word_vec is None or len(word_vec) == 0:
            return f"Word '{word}' has no vector representation."
        
        return f"Vector size for '{word}': {len(word_vec)} dimensions"

    @output
    @render.code
    @reactive.event(input.word_select)
    def word_vector():
        word = input.word_select()
        if not word:
            return "Select a word to see its vector."
        
        word_vec = nlp(word).vector
        if word_vec is None or len(word_vec) == 0:
            return f"Word '{word}' has no vector representation."
        
        return word_vec
    
    @output
    @render.text
    @reactive.event(input.run_embed)
    def vocab_sequence():
        words = get_words()
        if not words:
            return "No words entered. Please enter words and click Run."
        return " ".join(words)

    @output
    @render.data_frame
    @reactive.event(input.run_embed)
    def one_hot_table():
        words = get_words()
        if not words:
            return pd.DataFrame({"Message": ["No words entered. Please enter words and click Run."]})

        one_hot, _ = u2.convert_to_onehot(words, " ".join(words))
        df = pd.DataFrame(one_hot.to_numpy(),
                     index=one_hot.index,
                     columns=[f"{i}" for i in range(one_hot.shape[1])])
        view = (
            df.applymap(lambda x: f"{x:.0f}")
            .copy()
        )
        view.insert(0, "word", view.index) # word as first column for display reasons
        view.reset_index(drop=True, inplace=True)
        return render.DataGrid(view, height=400)
        return df
    
    @output
    @render.data_frame
    @reactive.event(input.run_one_hot_phrase)
    def one_hot_phrase_table():
        phrase = input.phrase().strip()
        if not phrase:
            return pd.DataFrame({"Message": ["No phrase entered. Please enter a phrase and click Run."]})
        
        words = get_words()
        one_hot, _ = u2.convert_to_onehot(words, phrase)
        df = pd.DataFrame(one_hot.to_numpy(),
                     index=one_hot.index,
                     columns=[f"{i}" for i in range(one_hot.shape[1])])
        view = (
            df.applymap(lambda x: f"{x:.0f}")
            .copy()
        )
        view.insert(0, "word", view.index)
        view.reset_index(drop=True, inplace=True)
        return render.DataGrid(view)


    @output
    @render.data_frame
    @reactive.event(input.run_embed)
    def embed_df():
        df = get_word_vecs()
        if df.empty:
            return pd.DataFrame({"Message": ["No data. Enter words and click Run."]})
        df = pd.DataFrame(df.to_numpy(),
                     index=get_words(),
                     columns=[f"D{i+1}" for i in range(df.shape[1])])


        view = (
            df.applymap(lambda x: f"{x:.3f}")
            .copy()
        )
        view.insert(0, "word", view.index)
        view.reset_index(drop=True, inplace=True)
        return render.DataGrid(view, height=400)

    @output
    @render.plot
    @reactive.event(input.embed_plot_btn)
    def embed_plot():
        df = get_word_vecs()
        words = get_words()
        if df is None or len(words) == 0:
            plt.figure()
            plt.text(0.5, 0.5, "Enter words and click Run", ha="center")
            return plt.gcf()

        reduced = u2.apply_pca(input.embed_dim(), df)
        
        if input.embed_dim() == 2:
            plot = u2.plot_word_embeddings_2d(reduced, words=words)
        else:
            plot = u2.plot_word_embeddings_3d(reduced, words=words)
        plt.title(f"Word Embeddings PCA (n_components={input.embed_dim()})")
        return plt.gcf()
    
    @output
    @render.ui
    @reactive.event(input.embed_plot_btn)
    def embedding_plot_2d():
        df_vecs = get_word_vecs()
        words = get_words()
        if df_vecs is None or df_vecs.empty or not words:
            return ui.HTML("<em>No embeddings to show. Enter words and click Run.</em>")

        emb = df_vecs.to_numpy()  # (N, D)
        labels = words  # color/legend = the word itself
        texts  = words  # hover text
        df = project_embeddings(emb, labels=labels, texts=texts, n_components=2)

        fig = go.Figure()
        # one trace per label for a clean legend
        for lbl, g in df.groupby("label", dropna=False):
            fig.add_trace(go.Scatter(
                x=g["x"], y=g["y"],
                mode="markers",
                name=str(lbl),
                text=g["text"],
                hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>%{text}<extra></extra>",
                marker=dict(size=8, opacity=0.9)
            ))

        fig.update_layout(
            xaxis_title="PC1", yaxis_title="PC2",
            margin=dict(l=40, r=20, t=30, b=40),
            showlegend=False, height=520
        )
        return ui.HTML(fig.to_html(full_html=False, include_plotlyjs=False, div_id="emb2d_div"))
    
    @output
    @render.ui
    @reactive.event(input.embed_plot_btn)
    def embedding_plot_3d():
        df_vecs = get_word_vecs()
        words = get_words()
        if df_vecs is None or df_vecs.empty or not words:
            return ui.HTML("<em>No embeddings to show. Enter words and click Run.</em>")

        emb = df_vecs.to_numpy()
        labels = words
        texts  = words
        df = project_embeddings(emb, labels=labels, texts=texts, n_components=3)

        fig = go.Figure()
        for lbl, g in df.groupby("label", dropna=False):
            fig.add_trace(go.Scatter3d(
                x=g["x"], y=g["y"], z=g["z"],
                mode="markers",
                name=str(lbl),
                text=g["text"],
                hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<br>%{text}<extra></extra>",
                marker=dict(size=4, opacity=0.9)
            ))

        fig.update_layout(
            scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
            margin=dict(l=0, r=0, t=30, b=0),
            showlegend=False, height=560
        )
        return ui.HTML(fig.to_html(full_html=False, include_plotlyjs=False, div_id="emb3d_div"))

app = App(app_ui, server, static_assets=Path(__file__).parent / "www")

if __name__ == "__main__":
    app.run()

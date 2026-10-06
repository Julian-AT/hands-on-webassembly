"""Exercise the unmodified Unit 2 controls on native or static Shiny."""

from repository import EVIDENCE
import re
import time
from playwright.sync_api import expect
from common import ROOT, PROOF


def unit2(page, url, prefix):
    page_errors = []
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    embedding_requests = []
    page.context.on(
        "request",
        lambda request: (
            embedding_requests.append(request.url)
            if "/assets/v1/embedding-runtime/" in request.url
            else None
        ),
    )
    start = time.perf_counter()
    page.goto(url, wait_until="domcontentloaded")
    app = page.frame_locator("iframe") if "/unit2/" in url else page
    app.locator("h2").wait_for(timeout=120000)
    app.locator("#img_op.shiny-bound-input").wait_for(state="attached", timeout=120000)
    startup = time.perf_counter() - start
    cases = {}
    app.locator("body").evaluate("""()=>{
      window.proofValues={};
      window.jQuery(document).on('shiny:value.proof shiny:error.proof',e=>window.proofValues[e.name]=(window.proofValues[e.name]||0)+1);
    }""")

    def fresh(id, action):
        before = app.locator("body").evaluate("(el,id)=>window.proofValues[id]||0", id)
        action()
        app.locator("body").evaluate(
            """(el,{id,before})=>new Promise((resolve,reject)=>{
          const end=Date.now()+60000;function poll(){
            if((window.proofValues[id]||0)>before)return resolve();
            if(Date.now()>end)return reject(new Error('No fresh '+id));setTimeout(poll,25);
          }poll();})""",
            {"id": id, "before": before},
        )

    def check(name, fn):
        try:
            detail = fn()
            errors = app.locator(".shiny-output-error:visible").all_text_contents()
            if errors:
                raise AssertionError(errors)
            cases[name] = {"status": "pass", "detail": detail}
        except Exception as e:
            cases[name] = {"status": "fail", "error": str(e)}
        print(prefix, name, cases[name]["status"], flush=True)

    def image_upload():
        app.locator("#img_file").set_input_files(
            str(ROOT / "assignments/2/resources/charlie_tiny.jpg")
        )
        expect(app.locator("#img_file_progress")).to_contain_text("Upload complete", timeout=30000)
        expect(app.locator("#img_plot img")).to_have_attribute(
            "src", re.compile("^data:image/"), timeout=60000
        )
        return app.locator("#image_dim").inner_text()

    check("image/upload", image_upload)
    for operation in (
        "channels",
        "rgba",
        "grayscale",
        "rotate",
        "flip",
        "crop",
        "blur",
        "hist",
        "segment",
        "show",
    ):

        def image_op():
            fresh("img_plot", lambda: app.locator("#img_op").select_option(operation))
            expect(app.locator("#img_plot img")).to_have_attribute(
                "src", re.compile("^data:image/"), timeout=60000
            )

        check("image/" + operation, image_op)
    app.get_by_role("tab", name="Signal Processing", exact=True).click()

    def generated():
        expect(app.locator("#audio_player audio")).to_be_visible(timeout=30000)
        expect(app.locator("#audio_player source")).to_have_attribute(
            "src", re.compile("^data:audio/wav;base64,")
        )
        return app.locator("#audio_player audio").evaluate(
            "el=>({duration:el.duration,readyState:el.readyState})"
        )

    check("audio/generated-player", generated)

    def playback():
        return app.locator("#audio_player audio").evaluate("""async audio=>{
          async function until(predicate,label){
            const deadline=Date.now()+15000;
            while(!predicate()){
              if(Date.now()>deadline)throw new Error(label);
              await new Promise(resolve=>setTimeout(resolve,25));
            }
          }
          await until(()=>Number.isFinite(audio.duration)&&audio.duration>0,'Audio metadata unavailable');
          audio.currentTime=0;
          await audio.play();
          await until(()=>audio.currentTime>0.1,'Playback clock did not advance');
          audio.pause();
          const target=Math.min(0.4,audio.duration/3);
          audio.currentTime=target;
          await until(()=>!audio.seeking&&Math.abs(audio.currentTime-target)<0.05,'Seek did not complete');
          return {duration:audio.duration,seekTarget:target,currentTime:audio.currentTime,paused:audio.paused};
        }""")

    check("audio/generated-playback-seeking", playback)
    app.locator("#view_mode").select_option("static")
    for kind, output in [
        ("spectrum", "spectrum_plot_static"),
        ("spectrogram", "spectrogram_plot_static"),
        ("wave", "waveform_plot_static"),
    ]:

        def signal():
            app.locator("#sig_plot").select_option(kind)
            expect(app.locator(f"#{output} img")).to_have_attribute(
                "src", re.compile("^data:image/"), timeout=60000
            )

        check("audio/" + kind, signal)

    def wav_upload():
        app.locator("#sig_op").select_option("upload")
        app.locator("#wav_file").set_input_files(
            str(ROOT / "assignments/2/resources/beethoven5_piano_10s.wav")
        )
        expect(app.locator("#wav_file_progress")).to_contain_text("Upload complete", timeout=30000)
        app.locator("#view_mode").select_option("player")
        expect(app.locator("#audio_player audio")).to_be_visible(timeout=60000)
        return app.locator("#audio_player audio").evaluate(
            "el=>({duration:el.duration,readyState:el.readyState})"
        )

    check("audio/wav-upload", wav_upload)
    check("audio/uploaded-playback-seeking", playback)
    if "/unit2/" in url:

        def deferred():
            assert not embedding_requests, (
                f"Embedding assets loaded before the text exercise: {embedding_requests}"
            )
            return {"requests_before_text_exercise": len(embedding_requests)}

        check("assets/embeddings-deferred", deferred)
    app.get_by_role("tab", name="Text Processing", exact=True).click()

    def words():
        app.locator("#word_list").fill(
            "king, queen, man, woman, Paris, London, cat, dog, Berlin, ice-cream"
        )
        fresh("one_hot_table", lambda: app.locator("#run_embed").click())
        expect(app.locator("#word_select option")).to_have_count(10, timeout=30000)
        return app.locator("#vocab_sequence").inner_text()

    check("text/vocabulary-one-hot", words)
    app.get_by_role("tab", name="Word Similarity", exact=True).click()

    def similarity():
        app.locator("#word_select1").select_option("king")
        app.locator("#word_select2").select_option("queen")
        fresh("word_similarity", lambda: app.locator("#find_similarity").click())
        return app.locator("#word_similarity").inner_text()

    check("text/similarity", similarity)

    def nearest():
        app.locator("#word_select_3").select_option("king")
        fresh("similar_words", lambda: app.locator("#find_similar").click())
        return app.locator("#similar_words").inner_text()

    check("text/nearest-distant", nearest)

    def arithmetic():
        app.locator("#word_computation").fill("king - man + woman")
        fresh("computed_words", lambda: app.locator("#compute_word").click())
        return app.locator("#computed_words").inner_text()

    check("text/arithmetic", arithmetic)
    app.get_by_role("tab", name="Plot Embeddings", exact=True).click()
    for dim in ("2d", "3d"):

        def projection():
            app.locator("#emb_view").select_option(dim)
            app.locator("#embed_plot_btn").click()
            expect(app.locator("#embedding_plot_" + dim + " .js-plotly-plot")).to_be_visible(
                timeout=60000
            )

        check("text/projection/" + dim, projection)
    if "/unit2/" in url:

        def complete_model():
            files = {url.rsplit("/", 1)[-1] for url in embedding_requests}
            assert files == {
                "manifest.json",
                "tokenizer.json",
                "word-rows.json.gz",
                "vectors.npy",
            }, files
            return {"requested_assets": sorted(files)}

        check("assets/complete-embedding-model", complete_model)
    try:
        page.screenshot(path=str(EVIDENCE / "screenshots" / f"{prefix}-text.png"))
    except Exception as e:
        cases["screenshot/" + str(len(cases))] = {"status": "fail", "error": str(e)}
    return {
        "startup_seconds": startup,
        "cases": cases,
        "page_errors": page_errors,
        "scope": "all ten image operations at defaults; generated/uploaded audio render, actual playback clock and seeking; text workflows. All control values, error recovery and pixel parity require separate coverage.",
    }

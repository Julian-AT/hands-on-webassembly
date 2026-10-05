"""Original Unit 4 workflows, used unchanged for native and browser apps."""
import re
import time
from playwright.sync_api import expect
from common import PROOF


def unit4(page, url, prefix):
    page.goto(url, wait_until='domcontentloaded')
    app = page.frame_locator('iframe') if '/unit4/' in url else page
    start = time.perf_counter()
    app.locator('h2').wait_for(timeout=120000)
    app.locator('#function.shiny-bound-input').wait_for(state='attached', timeout=120000)
    startup = time.perf_counter() - start
    app.locator('body').evaluate('''() => {
      window.proofValues = {};
      window.jQuery(document).on('shiny:value.proof shiny:error.proof', e => {
        window.proofValues[e.name] = (window.proofValues[e.name] || 0) + 1;
      });
    }''')
    cases = {}
    def tab(name):
        app.get_by_role('tab', name=name, exact=True).click()
    def plot(name):
        expect(app.locator(f'#{name} .js-plotly-plot')).to_be_visible(timeout=90000)
    def trigger(button, output):
        before = app.locator('body').evaluate('(el,id)=>window.proofValues[id]||0',output)
        app.locator('#'+button).click()
        app.locator('body').evaluate('''(el,{output,before})=>new Promise((resolve,reject)=>{
          const end=Date.now()+90000;
          function poll(){
            if((window.proofValues[output]||0)>before)return resolve();
            if(Date.now()>end)return reject(new Error('No fresh output: '+output));
            setTimeout(poll,25);
          } poll();
        })''',{'output':output,'before':before})
    def check(name, fn):
        t = time.perf_counter()
        try:
            detail = fn()
            errors = app.locator('.shiny-output-error:visible').all_text_contents()
            if errors: raise AssertionError(errors)
            cases[name] = {'status':'pass','seconds':time.perf_counter()-t,'detail':detail}
        except Exception as e:
            cases[name] = {'status':'fail','error':str(e)}
        print(prefix,name,cases[name]['status'],flush=True)
    def fit(function):
        tab('Function Fitting')
        app.locator('#function').select_option(function)
        trigger('fit','poly_mse_info')
        plot('poly_fit_plot')
        return app.locator('#poly_mse_info').inner_text()
    for function in ['Noisy sine','Mystery function']:
        check('fit/'+function,lambda f=function:fit(f))
    (PROOF/'evidence/screenshots').mkdir(exist_ok=True)
    try:
        page.screenshot(path=str(PROOF/'evidence/screenshots'/f'{prefix}-fitting.png'))
    except Exception as e:
        cases['screenshot/'+str(len(cases))]={'status':'fail','error':str(e)}
    for dataset, rows in [('Wine',178),('Breast Cancer',569),('Digits',535),('Pima Diabetes',768),('Iris',150),('Banknotes',1372)]:
        def load():
            tab('Data Settings & PCA'); tab('Summary Statistics')
            app.locator('#dataset').select_option(dataset)
            trigger('load','data_summary')
            expect(app.locator('#data_summary')).to_contain_text(f'Rows: {rows}',timeout=60000)
            text = app.locator('#data_summary').inner_text()
            tab('PCA'); plot('pca_scatter')
            return {'summary':text,'variance':app.locator('#pca_explained').inner_text()}
        check(dataset+'/load-pca',load)
        for classifier in ['k-NN','Decision Tree','Random Forest']:
            def classify():
                tab('Classification Models'); tab('Classification Report')
                app.locator('#classifier').select_option(classifier)
                trigger('apply_classifier','class_report')
                expect(app.locator('#class_report')).to_contain_text('precision',timeout=90000)
                report = app.locator('#class_report').inner_text()
                tab('Confusion Matrix'); plot('cm')
                tab('ROC / PR Curves'); plot('curve_plot')
                return report
            check(dataset+'/'+classifier,classify)
    try:
        page.screenshot(path=str(PROOF/'evidence/screenshots'/f'{prefix}-classification.png'))
    except Exception as e:
        cases['screenshot/'+str(len(cases))]={'status':'fail','error':str(e)}
    return {'startup_seconds':startup,'cases':cases,'scope':'two function defaults; all six datasets at default splits/PCA; three default classifiers with reports, confusion and ROC. Other controls are not certified.'}

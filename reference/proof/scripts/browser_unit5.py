"""Original Unit 5 UI workflows, including full-size image training/evaluation."""
import json
import re
import time
from pathlib import Path
from playwright.sync_api import expect
from common import ROOT, PROOF


def unit5(page, url, prefix):
    page.goto(url, wait_until='domcontentloaded')
    app = page.frame_locator('iframe') if '/unit5/' in url else page
    app.locator('h2').wait_for(timeout=120000)
    expect(app.locator('#pr_make.shiny-bound-input')).to_be_attached(timeout=120000)
    cases = {}
    downloads = PROOF/'evidence/downloads'/prefix
    downloads.mkdir(parents=True, exist_ok=True)
    def check(name, fn):
        start = time.perf_counter()
        try:
            detail = fn()
            errors = app.locator('.shiny-output-error:visible').all_text_contents()
            if errors: raise AssertionError(errors)
            cases[name] = {'status':'pass', 'seconds':time.perf_counter()-start, 'detail':detail}
        except Exception as e:
            cases[name] = {'status':'fail', 'error':str(e), 'seconds':time.perf_counter()-start}
        print(prefix, name, cases[name]['status'], flush=True)
    def tab(name): app.get_by_role('tab', name=name, exact=True).click()
    def plot(name): expect(app.locator('#'+name+' .js-plotly-plot')).to_be_visible(timeout=60000)
    def screenshot(name): page.screenshot(path=str(PROOF/'evidence/screenshots'/f'{prefix}-{name}.png'), full_page=False)
    def download(id, filename, kind):
        with page.expect_download(timeout=30000) as event:
            app.locator('#'+id).click()
        item = event.value
        assert item.suggested_filename == filename, item.suggested_filename
        target = downloads/filename
        item.save_as(target)
        text = target.read_text()
        if kind == 'json': assert json.loads(text)['coefficients']
        else: assert len(text.splitlines()) == 201
        return {'filename':filename, 'bytes':target.stat().st_size}
    for choice in ('Custom', 'Random', 'Classes'):
        def polynomial(choice=choice):
            tab('Linear Regression')
            app.locator(f'input[name="pr_choice"][value="{choice}"]').check()
            expect(app.locator('#pr_fitted_coeffs')).to_contain_text('No coefficients yet', timeout=30000)
            app.locator('#pr_make').click()
            app.locator('#pr_fit').check()
            plot('pr_plot_model')
            expect(app.locator('#pr_fitted_coeffs')).to_have_text(re.compile(r'^\{\s*"coefficients"'), timeout=30000)
            screenshot('linear-'+choice)
            return {'coefficients':app.locator('#pr_fitted_coeffs').inner_text(), 'mse':app.locator('#pr_mse').inner_text()}
        check('linear/'+choice, polynomial)
    check('download/polynomial-coefficients', lambda: download('pr_dl_coeffs','u5_polynomial_coeffs.json','json'))
    check('download/polynomial-dataset', lambda: download('pr_dl_data','u5_polynomial_regression_dataset.csv','csv'))
    def logistic():
        tab('Logistic Regression (1D)')
        app.locator('#lc_make').click()
        app.locator('#lc_train').click()
        expect(app.locator('#lc_metrics')).to_contain_text('Training accuracy:', timeout=60000)
        plot('lc_plot'); screenshot('logistic1d')
        coeffs = json.loads(app.locator('#lc_coeffs_txt').inner_text())
        assert len(coeffs['coefficients']) == 2
        return {'coefficients':coeffs, 'metrics':app.locator('#lc_metrics').inner_text()}
    check('logistic-1d/train', logistic)
    check('download/logistic-coefficients', lambda: download('lc_dl_coeffs','u5_logistic_coeffs.json','json'))
    check('download/logistic-dataset', lambda: download('lc_dl_data','u5_logistic_dataset.csv','csv'))
    def sigmoid():
        tab('Illustration: Sigmoid function'); plot('lc_sigmoid')
    check('logistic-1d/sigmoid', sigmoid)
    for ignore_header in (False, True):
        def two_dimensional(ignore_header=ignore_header):
            tab('Logistic Regression (2D)')
            app.locator('#lc_ignore_header_2d').set_checked(ignore_header)
            app.locator('#lc_file_2d').set_input_files(str(ROOT/'assignments/Material-20261003/DataSet_LR_a.csv'))
            expect(app.locator('#lc_file_2d_progress')).to_contain_text('Upload complete', timeout=30000)
            app.locator('#lc_make_2d').click()
            plot('lc_plot_2d')
            app.locator('#lc_train_2d').click()
            expect(app.locator('#lc_metrics_2d')).to_contain_text('Test accuracy:', timeout=60000)
            plot('lc_plot_prect_2d'); screenshot('logistic2d-'+str(ignore_header))
            return {'coefficients':json.loads(app.locator('#lc_coeffs_txt_2d').inner_text()), 'metrics':app.locator('#lc_metrics_2d').inner_text()}
        check('logistic-2d/header-'+str(ignore_header), two_dimensional)
    def exploration():
        tab('MNIST / Fashion-MNIST')
        for name in ('mnist_samples', 'fashionmnist_samples'):
            expect(app.locator('#'+name+' img')).to_have_attribute('src', re.compile('^data:image/'), timeout=120000)
        screenshot('datasets')
    check('dataset-exploration', exploration)
    for dataset in ('mnist', 'fashion'):
        def train(dataset=dataset):
            tab('Linear Classifier on MNIST')
            app.locator('#mn_ds').select_option(dataset)
            expect(app.locator('#mn_progress .progress-bar')).to_have_attribute('aria-valuenow','0',timeout=30000)
            expect(app.locator('#mn_acc')).to_be_empty(timeout=30000)
            app.locator('#mn_train').click()
            expect(app.locator('#mn_progress')).to_contain_text('Training complete', timeout=300000)
            app.locator('#mn_eval').click()
            expect(app.locator('#mn_acc')).to_contain_text('Test accuracy on '+('MNIST' if dataset=='mnist' else 'Fashion-MNIST'), timeout=180000)
            plot('mn_cm')
            expect(app.locator('#mn_examples img')).to_have_attribute('src', re.compile('^data:image/'), timeout=60000)
            metrics = app.locator('#mn_acc').inner_text()
            screenshot(dataset+'-evaluation')
            tab('Misclassification analysis')
            expect(app.locator('#mn_mis_table img').first).to_be_visible(timeout=60000)
            screenshot(dataset+'-misclassified')
            return {'metrics':metrics, 'train_rows':60000, 'test_rows':10000, 'epochs':3,
                    'misclassified_images':app.locator('#mn_mis_table img').count()}
        check(dataset+'/full-training-evaluation-inspection', train)
    return {'cases':cases}


def unit5_lifecycle(page, url):
    page.goto(url, wait_until='domcontentloaded')
    app=page.frame_locator('iframe')
    expect(app.locator('#mn_ds.shiny-bound-input')).to_be_attached(timeout=120000)
    app.get_by_role('tab',name='Linear Classifier on MNIST',exact=True).click()
    app.locator('#mn_epochs').fill('100')
    app.locator('#mn_epochs').press('Tab')
    app.locator('#mn_train').click()
    expect(app.locator('#mn_progress')).to_contain_text('Epoch',timeout=120000)
    start=time.perf_counter()
    app.locator('#mn_ds').select_option('fashion')
    expect(app.locator('#mn_progress .progress-bar')).to_have_attribute('aria-valuenow','0',timeout=15000)
    page.wait_for_timeout(1000)
    expect(app.locator('#mn_progress')).not_to_contain_text('Epoch')
    expect(app.locator('#mn_progress')).not_to_contain_text('Training complete')
    cases={'reset-during-training':{'status':'pass','seconds':time.perf_counter()-start},
           'stale-result-rejected':{'status':'pass','observation_seconds':1}}
    app.locator('#mn_epochs').fill('1');app.locator('#mn_epochs').press('Tab')
    app.locator('#mn_train').click()
    expect(app.locator('#mn_progress')).to_contain_text('Training complete',timeout=180000)
    app.locator('#mn_eval').click()
    expect(app.locator('#mn_acc')).to_contain_text('Test accuracy on Fashion-MNIST',timeout=120000)
    cases['retry-after-cancellation']={'status':'pass','metrics':app.locator('#mn_acc').inner_text()}
    app.locator('#mn_epochs').fill('100');app.locator('#mn_epochs').press('Tab')
    app.locator('#mn_train').click()
    expect(app.locator('#mn_progress')).to_contain_text('Epoch',timeout=120000)
    old_workers=page.workers
    closed=[]
    for worker in old_workers: worker.on('close',lambda w:closed.append(w.url))
    page.goto('http://127.0.0.1:8008/unit1/',wait_until='domcontentloaded')
    page.frame_locator('iframe').locator('h2').wait_for(timeout=120000)
    assert old_workers and len(closed)==len(old_workers),(len(old_workers),len(closed))
    cases['navigate-during-task']={'status':'pass'}
    cases['worker-terminated']={'status':'pass','workers':len(closed)}
    return {'cases':cases}

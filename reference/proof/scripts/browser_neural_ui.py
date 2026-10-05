"""Full-data loading and bounded original neural UI workflows."""
import ast
import json
import re
import time
from playwright.sync_api import expect
from common import PROOF,ROOT


def neural_ui(page,url,prefix,unit,fail_fast=False,on_case=None):
    start=time.perf_counter();page.goto(url,wait_until='domcontentloaded')
    app=page.frame_locator('iframe') if f'/unit{unit}/' in url else page
    app.locator('h2').wait_for(timeout=120000)
    cases={}
    def tab(name):app.get_by_role('tab',name=name,exact=True).click()
    def check(name,fn):
        try:
            detail=fn()
            errors=app.locator('.shiny-output-error:visible').all_text_contents()
            if errors:raise AssertionError(errors)
            cases[name]={'status':'pass','detail':detail}
        except Exception as e:cases[name]={'status':'fail','error':str(e)}
        if on_case is not None:on_case(name,cases[name])
        print(prefix,name,cases[name]['status'],flush=True)
        if fail_fast and cases[name]['status']!='pass':raise AssertionError(cases[name]['error'])
    def screenshot(name):page.screenshot(path=str(PROOF/'evidence/screenshots'/f'{prefix}-{name}.png'))
    if unit==6:
        tree=ast.parse((ROOT/'assignments/6/app.py').read_text())
        presets=ast.literal_eval(next(node.value for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='PRESETS' for target in node.targets)))
        for dataset,preset in [('toy_reg','Toy Regression – 1xLinear'),('toy_sine','Toy Regression – 2×Hidden (ReLU)'),('toy_bin','Toy Classification (Binary) – 1×Linear'),('toy_bin2','Toy Classification (Binary) – 2×Hidden (Tanh)'),('blob2d','Toy Classification (Binary) – 2×Hidden (ReLU)'),('moons2d','Toy Classification (Binary) – 2×Hidden (ReLU)')]:
            def train():
                tab('FNN: Data');app.locator('#dataset').select_option(dataset);app.locator('#load').click()
                expect(app.locator('#sample_plot img')).to_have_attribute('src',re.compile('^data:image/'),timeout=60000)
                tab('FNN:Architecture');app.locator('#preset').select_option(preset);app.locator('#load_preset').click()
                expected=json.dumps(presets[preset],indent=2)
                expect(app.locator('#arch_text')).to_have_value(expected,timeout=30000)
                app.locator('body').evaluate('''(el,expected)=>new Promise((resolve,reject)=>{
                  const end=Date.now()+30000;
                  function poll(){if(window.Shiny.shinyapp.$inputValues.arch_text===expected)return resolve();if(Date.now()>end)return reject(new Error('Selected preset has not reached the reactive input'));setTimeout(poll,25)}poll();
                })''',expected)
                # Preset-to-editor synchronization is asynchronous. The summary
                # must refer to the selected architecture before training.
                expect(app.locator('#model_summary')).to_contain_text('Total parameters',timeout=30000)
                app.locator('#apply_arch').click()
                tab('FNN: Training');app.locator('#reset').click()
                expect(app.locator('#best_model_info')).to_contain_text('not set',timeout=30000)
                app.locator('#train').click()
                expect(app.locator('#train_progress')).to_contain_text('Training complete',timeout=120000)
                expect(app.locator('#best_model_info')).to_contain_text('Using last epoch model',timeout=30000)
                result=app.locator('#best_model_info').inner_text()
                tab('FNN: Predict');expect(app.locator('#pred_plot img')).to_have_attribute('src',re.compile('^data:image/'),timeout=60000)
                return result
            check('toy-training/'+dataset,train)
        for variant in ('MNIST','FashionMNIST'):
            def load():
                tab('FNN: Data');app.locator('#dataset').select_option(variant);app.locator('#load').click()
                expect(app.locator('#data_info')).to_contain_text('54000',timeout=120000)
                expect(app.locator('#data_info')).to_contain_text('10000',timeout=30000)
                expect(app.locator('#sample_plot img')).to_have_attribute('src',re.compile('^data:image/'),timeout=30000)
                return app.locator('#data_info').inner_text()
            check('full-data/'+variant,load)
    else:
        def filters():
            tab('Filters');app.locator('#conv_image').set_input_files(str(ROOT/'assignments/2/resources/charlie_tiny.jpg'))
            expect(app.locator('#conv_image_progress')).to_contain_text('Upload complete',timeout=30000)
            app.locator('#apply_filter').click()
            expect(app.locator('#convolution_result img')).to_have_attribute('src',re.compile('^data:image/'),timeout=60000)
        check('filters/upload-sobel',filters)
        for variant in ('MNIST','FashionMNIST','CIFAR10','SVHN','USPS'):
            def load():
                tab('CNN: Data')
                plot=app.locator('#sample_plot img')
                previous=plot.get_attribute('src') if plot.count() else None
                app.locator('#variant').select_option(variant);app.locator('#load_data').click()
                expect(app.locator('#data_info')).to_contain_text('Loaded '+variant+' dataset.',timeout=120000)
                train,test={'MNIST':(54000,10000),'FashionMNIST':(54000,10000),'CIFAR10':(45000,10000),'SVHN':(65931,26032),'USPS':(6562,2007)}[variant]
                expect(app.locator('#data_info')).to_contain_text(f'training set:   {train}',timeout=120000)
                expect(app.locator('#data_info')).to_contain_text(f'test set:       {test}',timeout=30000)
                expect(plot).to_have_attribute('src',re.compile('^data:image/'),timeout=30000)
                if previous:expect(plot).not_to_have_attribute('src',previous,timeout=30000)
                return app.locator('#data_info').inner_text()
            check('full-data/'+variant,load)
    screenshot('final')
    return {'cases':cases,'startup_seconds':time.perf_counter()-start,'scope':'bounded original UI paths; full preset image training and numerical histories remain separate requirements'}

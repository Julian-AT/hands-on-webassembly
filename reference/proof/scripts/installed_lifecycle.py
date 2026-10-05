"""One page interface for installed browsers, with no diagnostic injection."""
from contextlib import contextmanager
import time


class WebDriverPage:
    def __init__(self, driver, parent):
        self.driver, self.parent = driver, parent

    def goto(self, url, **unused): self.driver.get(url)
    def reload(self, **unused): self.driver.refresh()

    def evaluate(self, expression):
        result = self.driver.execute_async_script('''
const done=arguments[arguments.length-1];
Promise.resolve().then(()=>{
  const value=(''' + expression + ''');
  return typeof value==='function'?value():value;
}).then(value=>done({value:value===undefined?null:value}),error=>done({error:String(error)}));
''')
        if 'error' in result: raise RuntimeError(result['error'])
        return result['value']

    def wait_for_function(self, expression, *, timeout):
        deadline = time.monotonic() + timeout/1000
        while time.monotonic() < deadline:
            if self.evaluate(expression): return
            time.sleep(.1)
        raise TimeoutError(f'Installed browser readiness deadline exceeded ({timeout} ms)')

    def close(self):
        self.driver.close()
        self.driver.switch_to.window(self.parent)


class WebDriverContext:
    def __init__(self, driver): self.driver = driver
    def new_page(self):
        parent = self.driver.current_window_handle
        self.driver.switch_to.new_window('tab')
        return WebDriverPage(self.driver, parent)


@contextmanager
def installed_browser(name):
    if name in ('chrome','edge'):
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(channel='chrome' if name=='chrome' else 'msedge')
            try:
                context = browser.new_context(viewport={'width':1440,'height':1000})
                yield context, browser.version, dict(backend='playwright',channel=name)
            finally:
                browser.close()
    elif name in ('firefox','safari'):
        from selenium import webdriver
        if name == 'firefox':
            options = webdriver.FirefoxOptions()
            options.binary_location = '/Applications/Firefox.app/Contents/MacOS/firefox'
            options.add_argument('-headless')
            driver = webdriver.Firefox(options=options)
        else:
            driver = webdriver.Safari()
        try:
            driver.set_window_size(1440,1000)
            driver.set_page_load_timeout(120)
            driver.set_script_timeout(45)
            yield WebDriverContext(driver), driver.capabilities['browserVersion'], dict(
                backend='selenium', capabilities=driver.capabilities,
                binary='/Applications/Firefox.app/Contents/MacOS/firefox' if name=='firefox' else '/usr/bin/safaridriver')
        finally:
            driver.quit()
    else:
        raise ValueError('Only installed Chrome, Edge, Firefox and Safari may certify lifecycle')

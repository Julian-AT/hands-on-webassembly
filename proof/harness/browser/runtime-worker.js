importScripts('../shinylive/pyodide/pyodide.js');
self.onmessage=async()=>{try{
  const py=await loadPyodide({indexURL:'../shinylive/pyodide/'});
  await py.loadPackage(['numpy','matplotlib','pandas','scipy','scikit-learn','plotly','course-forest-criterion']);
  const result=await py.runPythonAsync(`
import json,io,numpy,pandas,scipy,sklearn,matplotlib,plotly
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ft2font as ft
from forest_criterion import fused_proxy
expected={'numpy':'1.26.4','pandas':'2.2.3','scipy':'1.14.1','sklearn':'1.6.1','matplotlib':'3.10.8','plotly':'7.1.0'}
versions={name:globals()[name].__version__ for name in expected}
checks={'scientific-versions':{'status':'pass' if versions==expected else 'fail','versions':versions}}
checks['freetype-version']={'status':'pass' if ft.__freetype_version__=='2.6.1' else 'fail','version':ft.__freetype_version__}
f,a=plt.subplots();a.plot([0,1],[1,0]);out=io.BytesIO();f.savefig(out,format='png');plt.close(f)
checks['matplotlib-agg']={'status':'pass' if len(out.getvalue())>1000 else 'fail','png_bytes':len(out.getvalue())}
a=fused_proxy(119,1-(6**2+113**2)/119**2,11,1-(9**2+2**2)/11**2)
b=fused_proxy(11,1-(9**2+2**2)/11**2,119,1-(6**2+113**2)/119**2)
checks['native-fused-gini-score']={'status':'pass' if a>b else 'fail','scores':[a,b]}
json.dumps({'checks':checks})
`);
  self.postMessage({done:true,...JSON.parse(result)});
}catch(error){self.postMessage({error:String(error)});}};

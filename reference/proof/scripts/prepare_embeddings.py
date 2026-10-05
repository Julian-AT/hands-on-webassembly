"""Export actual model rules, all vector keys, and English regression fixtures."""
import ast
import gzip
import json
from pathlib import Path
import sys
import numpy as np
import spacy
from spacy.attrs import ORTH
from spacy.tokenizer import Tokenizer
from common import PROOF,sha,write_json


def main():
    nlp=spacy.load('en_core_web_md')
    assert spacy.__version__=='3.7.5' and nlp.meta['version']=='3.7.1'
    tok=nlp.tokenizer
    raw=Tokenizer(nlp.vocab,rules={},prefix_search=tok.prefix_search,suffix_search=tok.suffix_search,infix_finditer=tok.infix_finditer,token_match=tok.token_match,url_match=tok.url_match)
    root=PROOF/'assets/v1/embedding-runtime'
    root.mkdir(parents=True,exist_ok=True)
    import importlib.metadata
    import shutil
    license_path=Path(importlib.metadata.distribution('spacy').locate_file('spacy-3.7.5.dist-info/LICENSE'))
    shutil.copy2(license_path,root/'SPACY-LICENSE')
    import en_core_web_md
    model_root=Path(en_core_web_md.__file__).parent/'en_core_web_md-3.7.1'
    for name in ('LICENSE','LICENSES_SOURCES'):
        shutil.copy2(model_root/name,root/('MODEL-'+name))
    config={name:getattr(tok,attr).__self__.pattern if getattr(tok,attr) else None for name,attr in [('prefix','prefix_search'),('suffix','suffix_search'),('infix','infix_finditer'),('token','token_match'),('url','url_match')]}
    config['rules']={key:[item[ORTH] for item in rules] for key,rules in tok.rules.items()}
    config['special_patterns']=[([t.text for t in raw(key)],key) for key in tok.rules if not tok.faster_heuristics or tok.find_prefix(key) or tok.find_suffix(key) or tok.find_infix(key) or ' ' in key]
    write_json(root/'tokenizer.json',config)
    rows=[(nlp.vocab.strings[key],row) for key,row in nlp.vocab.vectors.key2row.items()]
    (root/'word-rows.json.gz').write_bytes(gzip.compress(json.dumps(rows,ensure_ascii=False,separators=(',',':')).encode(),mtime=0))
    np.save(root/'vectors.npy',nlp.vocab.vectors.data,allow_pickle=False)
    write_json(root/'manifest.json',{'spacy':spacy.__version__,'model':nlp.meta['version'],'keys':len(rows),'vector_shape':list(nlp.vocab.vectors.data.shape),'files':{name:sha(root/name) for name in ('tokenizer.json','word-rows.json.gz','vectors.npy')}})
    corpus=set(tok.rules)
    tests=Path(spacy.__file__).parent/'tests/lang/en'
    source_tests={}
    for file in sorted(tests.glob('test*.py')):
        if any(name in file.name for name in ('token','exceptions','prefix','punct','text','indices')):
            source_tests[file.name]=sha(file)
            corpus.update(n.value for n in ast.walk(ast.parse(file.read_text())) if isinstance(n,ast.Constant) and isinstance(n.value,str) and len(n.value)<10000)
    for word in ('king','King','KING',"can't",'U.S.A.','ice-cream','New York','café','👋','zzqvnotaword','cat zzqvnotaword'):
        for before in ('',' ','   ','\n','\t','(','“'):
            for after in ('',' ','  ','\r\n',', world!',')','”'):corpus.add(before+word+after)
    corpus.update(['','   ','a\u00a0b','a\u2003b','👩🏽‍💻','a\x1fb','naïve café\n東京'])
    sys.path.insert(0,str(PROOF/'runtime'))
    from embedding_runtime import English
    browser=English(root)
    failures=[];vectors=[];fixtures=[]
    ordered=sorted(corpus)
    for i,text in enumerate(ordered):
        doc=nlp.make_doc(text);other=browser(text)
        expected=[(t.text,t.idx,t.text_with_ws) for t in doc]
        actual=[(t.text,t.idx,t.text_with_ws) for t in other]
        if actual!=expected or not np.allclose(doc.vector,other.vector,rtol=1e-4,atol=1e-5):
            failures.append({'text':text,'expected':expected,'actual':actual})
        other_text=ordered[(i+1)%len(ordered)]
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            similarity=float(doc.similarity(nlp.make_doc(other_text)))
        fixtures.append({'text':text,'tokens':expected,'norm':float(doc.vector_norm),'other_text':other_text,'similarity':similarity})
        vectors.append(doc.vector)
    write_json(PROOF/'evidence/embedding-regression.json',{'status':'fail' if failures else 'pass','cases':len(fixtures),'failures':failures,'source_tests':source_tests,'qualification':'Every string literal from the installed English tokenizer test modules, all model exceptions and generated punctuation/whitespace cases compared against the original loaded tokenizer; not a claim to run custom-tokenizer tests unchanged.'})
    write_json(PROOF/'site/probes/embedding-cases.json',fixtures)
    np.save(PROOF/'site/probes/embedding-vectors.npy',np.stack(vectors),allow_pickle=False)
    print('embedding cases',len(fixtures),'failures',len(failures))
    if failures:print(json.dumps(failures[:8],indent=2));raise SystemExit(1)


if __name__=='__main__':main()

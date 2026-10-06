"""Pinned English tokenizer/document-vector surface used by Assignment 2.

Affix/infix ordering follows spaCy 3.7.5 tokenizer.pyx (MIT); regexes, exceptions,
retokenization patterns and every vocabulary/vector entry come from the actual
loaded en_core_web_md 3.7.1 model. No linguistic pipeline is exposed or emulated.
"""
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace
import numpy as np

__version__='3.7.5'


class Token:
    def __init__(self,text,idx=0,space=False):self.text,self.idx,self.space=text,idx,space
    @property
    def text_with_ws(self):return self.text+(' ' if self.space else '')
    def __str__(self):return self.text


class Doc:
    def __init__(self,nlp,text,tokens):self.nlp,self.text,self.tokens=nlp,text,tokens;self._vector=None
    def __iter__(self):return iter(self.tokens)
    def __len__(self):return len(self.tokens)
    def __getitem__(self,key):return self.tokens[key]
    def __str__(self):return self.text
    @property
    def vector(self):
        if self._vector is None:
            zero=np.zeros(300,dtype=np.float32)
            vectors=[self.nlp.vectors[self.nlp.rows[t.text]] if t.text in self.nlp.rows else zero for t in self.tokens]
            self._vector=sum(vectors)/len(vectors) if vectors else zero
        return self._vector
    @property
    def vector_norm(self):return math.sqrt(sum(float(v*v) for v in self.vector))
    def similarity(self,other):
        if [t.text for t in self]==[t.text for t in other]:return 1.0
        norm=self.vector_norm*other.vector_norm
        return float(np.dot(self.vector,other.vector)/norm) if norm else 0.0


class English:
    def __init__(self,root=None):
        self.root=Path(root) if root else Path(__file__).parent/'embedding-assets'
        self.manifest=json.loads((self.root/'manifest.json').read_text())
        self._vectors=None
        self._rows=None
        self._tokenizer_loaded=False

    def _read(self,name):
        path=self.root/name
        if path.exists():
            data=path.read_bytes()
        else:
            from js import location,XMLHttpRequest,Uint8Array
            from urllib.parse import urljoin
            request=XMLHttpRequest.new()
            request.open('GET',urljoin(str(location.href),'../assets/v1/embedding-runtime/'+name),False)
            request.responseType='arraybuffer'
            request.send()
            if request.status!=200:
                raise OSError(f'Embedding asset request failed ({request.status}): {name}. Retry the exercise.')
            data=Uint8Array.new(request.response).to_py().tobytes()
        if hashlib.sha256(data).hexdigest()!=self.manifest['files'][name]:
            raise ValueError(f'Embedding checksum mismatch: {name}. Retry the exercise.')
        return data

    def _load_vectors(self):
        if self._vectors is not None:return
        # Publish together only after all reads and checks pass; failed fetches
        # leave the model retryable rather than retaining a partial vocabulary.
        rows=dict(json.loads(gzip.decompress(self._read('word-rows.json.gz'))))
        vectors=np.load(io.BytesIO(self._read('vectors.npy')),allow_pickle=False)
        if len(rows)!=self.manifest['keys'] or list(vectors.shape)!=self.manifest['vector_shape']:
            raise ValueError('Embedding asset dimensions do not match the complete model')
        self._rows,self._vectors=rows,vectors

    @property
    def vectors(self):
        self._load_vectors()
        return self._vectors

    @property
    def rows(self):
        self._load_vectors()
        return self._rows

    def _load_tokenizer(self):
        if self._tokenizer_loaded:return
        config=json.loads(self._read('tokenizer.json'))
        self.rules=config['rules']
        for name in ('prefix','suffix','infix','token','url'):
            pattern=config[name]
            setattr(self,name,re.compile(pattern) if pattern else None)
        self.special_patterns={}
        for pattern,key in config['special_patterns']:
            self.special_patterns.setdefault(pattern[0],[]).append((pattern,key))
        self._tokenizer_loaded=True
    def segment(self,string):
        self._load_tokenizer()
        if string in self.rules:return self.rules[string][:]
        prefixes,suffixes=[],[]
        last=0
        while string and len(string)!=last:
            if self.token and self.token.match(string) or string in self.rules:break
            last=len(string)
            pre=self.prefix.search(string) if self.prefix else None
            pre_len=pre.end()-pre.start() if pre else 0
            if pre_len:
                prefix,minus_pre=string[:pre_len],string[pre_len:]
                if minus_pre and minus_pre in self.rules:
                    string=minus_pre;prefixes.append(prefix);break
            suf=self.suffix.search(string[pre_len:]) if self.suffix else None
            suf_len=suf.end()-suf.start() if suf else 0
            if suf_len:
                suffix,minus_suf=string[-suf_len:],string[:-suf_len]
                if minus_suf and minus_suf in self.rules:
                    string=minus_suf;suffixes.append(suffix);break
            if pre_len and suf_len and pre_len+suf_len<=len(string):
                string=string[pre_len:-suf_len];prefixes.append(prefix);suffixes.append(suffix)
            elif pre_len:string=minus_pre;prefixes.append(prefix)
            elif suf_len:string=minus_suf;suffixes.append(suffix)
        middle=[]
        if string:
            if string in self.rules:middle=self.rules[string][:]
            elif self.token and self.token.match(string) or self.url and self.url.match(string):middle=[string]
            else:
                start=0
                for match in self.infix.finditer(string) if self.infix else []:
                    a,b=match.span()
                    if a==0:continue
                    if a!=start:middle.append(string[start:a])
                    if a!=b:middle.append(string[a:b])
                    start=b
                if string[start:]:middle.append(string[start:])
        return prefixes+middle+suffixes[::-1]
    def __call__(self,text):
        if not isinstance(text,str):raise TypeError('Expected text')
        if not text:return Doc(self,text,[])
        if len(text)>=2**30:raise ValueError('Text exceeds tokenizer limit')
        self._load_tokenizer()
        tokens=[]
        def append(span,start):
            offset=start
            for word in self.segment(span):
                tokens.append(Token(word,offset));offset+=len(word)
        start=0;in_ws=text[0].isspace()
        for i,char in enumerate(text):
            if char.isspace()!=in_ws:
                if start<i:append(text[start:i],start)
                if char==' ':
                    if tokens:tokens[-1].space=True
                    start=i+1
                else:start=i
                in_ws=not in_ws
        if start<len(text):append(text[start:],start)
        # Match exception phrases on ORTH, longest first, earliest on ties.
        matches=[]
        for start,t in enumerate(tokens):
            for pattern,key in self.special_patterns.get(t.text,[]):
                end=start+len(pattern)
                if [x.text for x in tokens[start:end]]==pattern:matches.append((start,end,key))
        seen=set();chosen={}
        for start,end,key in sorted(matches,key=lambda x:(-(x[1]-x[0]),x[0])):
            if start not in seen and end-1 not in seen:chosen[start]=(end,key)
            seen.update(range(start,end))
        output=[];i=0
        while i<len(tokens):
            if i not in chosen:output.append(tokens[i]);i+=1;continue
            end,key=chosen[i]
            actual=text[tokens[i].idx:tokens[end-1].idx+len(tokens[end-1].text)]
            if actual in self.rules:
                offset=tokens[i].idx
                for word in self.rules[actual]:output.append(Token(word,offset));offset+=len(word)
                output[-1].space=tokens[end-1].space
            else:output.extend(tokens[i:end])
            i=end
        return Doc(self,text,output)


lang=SimpleNamespace(en=SimpleNamespace(English=English))

def load(name):
    if name!='en_core_web_md':raise ValueError('Only the pinned course model is supported')
    return English()

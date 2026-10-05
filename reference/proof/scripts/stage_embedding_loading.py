"""Source-checked asynchronous embedding preparation, preserving the original UI."""
import ast
import hashlib
import json
from common import PROOF,sha


def adapter(source):
    start=source.index('        else:\n            from js import location,XMLHttpRequest,Uint8Array')
    end=source.index('\n        if hashlib.sha256(data)',start)
    source=source[:start]+"        else:\n            raise RuntimeError('Embedding model is not prepared. Load the vocabulary and retry.')"+source[end:]
    marker='    def _load_vectors(self):'
    method='''    def install_prepared(self,tokenizer,rows,vectors):
        if len(rows)!=self.manifest['keys'] or list(vectors.shape)!=self.manifest['vector_shape'] or vectors.dtype!=np.float32:
            raise ValueError('The complete embedding model is required')
        temporary=English(self.root)
        temporary._read=lambda name:tokenizer.encode('utf-8')
        temporary._load_tokenizer()
        values={name:getattr(temporary,name) for name in
                ('rules','prefix','suffix','infix','token','url','special_patterns','_tokenizer_loaded')}
        values.update(_rows=rows,_vectors=vectors)
        self.__dict__.update(values)

'''
    return source.replace(marker,method+marker,1)


def application(source):
    source=source.replace('import embedding_runtime as spacy\n','import embedding_runtime as spacy\nimport embedding_preload\n',1)
    marker='    # Word Embeddings\n'
    preparation='''    # Disposable complete embedding preparation, using a vocabulary snapshot.
    from shiny import req as _course_req
    import json as _course_json
    from pathlib import Path as _CoursePath
    _course_embedding_config=_course_json.loads(_CoursePath(__file__).with_name('course-embedding.json').read_text())
    _course_embedding_owner=embedding_preload.Owner(nlp,_course_embedding_config)
    _course_embedding_ticket=reactive.Value(0)

    @reactive.extended_task
    async def _course_prepare_embeddings(ticket,words):
        if any(word.strip() for word in words.split(',')):
            await _course_embedding_owner.prepare()
        return dict(ticket=ticket,words=words)

    @reactive.effect
    @reactive.event(input.run_embed)
    def _course_start_embedding_preparation():
        _course_embedding_owner.cancel()
        _course_prepare_embeddings.cancel()
        ticket=_course_embedding_ticket.get()+1
        _course_embedding_ticket.set(ticket)
        _course_prepare_embeddings.invoke(ticket,input.word_list())

    @reactive.calc
    def _course_embedding_ready():
        receipt=_course_prepare_embeddings.result()
        _course_req(receipt['ticket']==_course_embedding_ticket.get())
        return receipt

    @reactive.effect
    @reactive.event(input.word_list)
    def _course_replace_embedding_preparation():
        if _course_prepare_embeddings.status()=='running':
            _course_embedding_ticket.set(_course_embedding_ticket.get()+1)
            _course_embedding_owner.cancel()
            _course_prepare_embeddings.cancel()

    def _course_close_embedding_preparation():
        _course_embedding_owner.cancel()
        _course_prepare_embeddings.cancel()
    session.on_ended(_course_close_embedding_preparation)

'''
    if source.count(marker)!=1:raise ValueError('Embedding section source changed')
    source=source.replace(marker,preparation+marker)
    tree=ast.parse(source);lines=source.splitlines(keepends=True)
    insertions=[]
    guarded={'get_word_vecs','similar_words','computed_words','word_similarity','word_vector_size','word_vector'}
    rendered={'embed_df','embed_plot','embedding_plot_2d','embedding_plot_3d',*guarded}
    for node in ast.walk(tree):
        if not isinstance(node,ast.FunctionDef) or node.name not in rendered:continue
        for decorator in node.decorator_list:
            if isinstance(decorator,ast.Call) and isinstance(decorator.func,ast.Attribute) and decorator.func.attr=='event':
                line=lines[decorator.lineno-1]
                insertions.append((decorator.lineno-1,line.replace(')',', _course_prepare_embeddings.status)',1),True))
        if node.name in guarded:
            first=next(statement for statement in node.body if any(isinstance(call,ast.Call) and
                ((isinstance(call.func,ast.Name) and call.func.id=='nlp') or
                 (isinstance(call.func,ast.Attribute) and call.func.attr=='get_word_vectors')) for call in ast.walk(statement)))
            insertions.append((first.lineno-1,'        _course_embedding_ready()\n',False))
    for index,text,replace in sorted(insertions,reverse=True):
        if replace:lines[index]=text
        else:lines.insert(index,text)
    return ''.join(lines)


def write_embedding_config(stage,source):
    source_id=hashlib.sha256(source.encode()).hexdigest()
    inputs={str(p.relative_to(PROOF)):sha(p) for p in (PROOF/'runtime/embedding_preload.py',
        PROOF/'runtime/course-embedding-worker.js',PROOF/'runtime/embedding_runtime.py',
        PROOF/'assets/v1/embedding-runtime/manifest.json')}
    build_id=hashlib.sha256(json.dumps(dict(inputs=inputs,source_id=source_id),sort_keys=True).encode()).hexdigest()
    (stage/'course-embedding.json').write_text(json.dumps(dict(build_id=build_id,source_id=source_id,
        manifest_sha256=sha(PROOF/'assets/v1/embedding-runtime/manifest.json'))))
    return inputs,build_id,source_id

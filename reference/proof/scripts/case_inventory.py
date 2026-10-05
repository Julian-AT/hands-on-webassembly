"""Stable behavior-case identities, independent of application source lines.

This adds requirements to the existing source-line contract. It does not infer
passing evidence from DOM presence, and it does not remove legacy requirements.
Dynamic choices require runtime expansion before exhaustive certification.
"""
import ast
import hashlib
from urllib.parse import quote
from functools import lru_cache
from common import ROOT,PROOF,UNITS,write_json


def literal(node):
    try:return ast.literal_eval(node)
    except (ValueError,TypeError):return None


@lru_cache(maxsize=7)
def tab_paths(unit):
    result={}
    def visit(node,path):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='nav_panel':
            label=literal(node.args[0]) if node.args else None
            if isinstance(label,str):path=(*path,label)
            result[(node.lineno,ast.unparse(node))]=path
        for child in ast.iter_child_nodes(node):visit(child,path)
    visit(ast.parse((ROOT/UNITS[unit]/'app.py').read_text()),())
    return result


def identities(unit,interface):
    cases={}
    for item in interface:
        call=ast.parse(item['args'],mode='eval').body
        keywords={k.arg:k.value for k in call.keywords if k.arg}
        identifier=literal(keywords.get('id'))
        if identifier is None and call.args:identifier=literal(call.args[0])
        if not isinstance(identifier,(str,int)):
            # No line number participates in an ID. Keep the original expression
            # so a reviewer can explicitly expand every dynamic declaration.
            identifier='expression-'+hashlib.sha256(ast.dump(call).encode()).hexdigest()[:16]
        if item['call']=='nav_panel':
            ancestry=tab_paths(int(unit)).get((item['line'],item['args']))
            if ancestry:identifier='/'.join(ancestry)
        kind='tab' if item['call'].startswith('nav_') else 'output' if item['call'].startswith('output_') else 'download' if 'download' in item['call'] else 'control'
        base=f'unit{unit}/{kind}/{quote(str(identifier),safe="-_:")}'
        scenarios={'default-behavior'}
        choices=literal(keywords.get('choices'))
        if choices is None and item['call'] in ('input_select','input_selectize','input_radio_buttons') and len(call.args)>2:
            choices=literal(call.args[2])
        if isinstance(choices,(dict,list,tuple)):
            scenarios.update('choice-'+quote(str(choice),safe='-_') for choice in choices)
        if item['call']=='input_checkbox':scenarios.update(('enabled','disabled'))
        if item['call'] in ('input_slider','input_numeric'):
            for boundary in ('min','max'):
                if boundary in keywords or item['call']=='input_slider':scenarios.add('boundary-'+boundary)
        if item['call']=='input_file':scenarios.update(('valid-upload','invalid-format','interrupted-upload','retry-upload'))
        if kind=='download':scenarios.add('filename-and-roundtrip-content')
        if kind=='output':scenarios.add('fresh-result-after-input-change')
        dynamic=(not isinstance(choices,(dict,list,tuple)) or not choices) and item['call'] in ('input_select','input_selectize','input_radio_buttons')
        for scenario in sorted(scenarios):
            case_id=base+'/'+scenario
            record=cases.setdefault(case_id,dict(id=case_id,unit=unit,identifier=str(identifier),kind=kind,
                scenario=scenario,declarations=[],requires_runtime_expansion=dynamic,
                assertion='Assert the resulting value, content, computation, workflow or recoverable error; element existence is insufficient.'))
            record['declarations'].append({'call':item['call'],'source_line':item['line']})
            record['requires_runtime_expansion'] |= dynamic
    return list(cases.values())


def main():
    import json
    inventory=json.loads((PROOF/'evidence/source-inventory.json').read_text())
    cases=[case for unit,item in inventory.items() for case in identities(unit,item['interface'])]
    write_json(PROOF/'evidence/behavior-case-inventory.json',dict(status='inventory',cases=cases,
        required_cases=len(cases),dynamic_declarations_require_expansion=True,
        scope='Additional stable behavior requirements. Executor/evidence coverage is pending; this inventory is not a test pass.'))
    print(len(cases),'stable behavior cases')


if __name__=='__main__':main()

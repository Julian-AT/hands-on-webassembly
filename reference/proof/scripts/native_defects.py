"""Record confirmed native defects separately; never silently repair originals."""
import json
import sys
import pandas as pd
import torch
from common import ROOT,PROOF,extract,sha,write_json


def main():
    sys.path.insert(0,str(ROOT/'assignments/2'))
    import u2_utils
    cases={}
    words=['New York','cat']
    try:
        # Exact two operations in original one_hot_table (including its labels).
        one_hot,_=u2_utils.convert_to_onehot(words,' '.join(words))
        pd.DataFrame(one_hot.to_numpy(),index=words,columns=[str(i) for i in range(one_hot.shape[1])])
    except ValueError as e:cases['unit2/multiword-one-hot']={'status':'confirmed','error':str(e),'source_sha256':sha(ROOT/'assignments/2/app.py')}
    scope={'json':json,'nn':torch.nn}
    exec(extract(6,['ACTIVATIONS','parse_architecture']),scope)
    try:scope['parse_architecture']('[{"type":"linear","out_features":1}]')
    except AttributeError as e:cases['unit6/top-level-architecture-list']={'status':'confirmed','error':str(e),'source_sha256':sha(ROOT/'assignments/6/app.py')}
    write_json(PROOF/'evidence/native-defects.json',cases)
    print(json.dumps(cases,indent=2))


if __name__=='__main__':main()

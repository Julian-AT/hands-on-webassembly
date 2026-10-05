import os
import shutil
import numpy as np
from common import PROOF, extract, write_json, sha


def main():
    target = PROOF / "site/probes"
    definitions = extract(1,["load_wine","load_penguins","load_iris_df","load_breast_cancer_df","class_labels"])
    from training import nested
    definitions += "\n" + "\n".join(nested(1,name) for name in ("get_data","get_target_column","data_summary","reduction_plot","analysis_plot"))
    (target / "tabular-definitions.py").write_text(definitions)
    shutil.copy2(PROOF / "runtime/tabular_probe.py",target / "tabular_probe.py")
    shutil.copy2(PROOF / "assets/v1/penguins.csv",target / "penguins.csv")
    import sys
    sys.path.insert(0,str(PROOF / "runtime"))
    from tabular_probe import run_tabular
    shutil.copy2(PROOF / "runtime/tabular_regressions.py",target / "tabular_regressions.py")
    os.chdir(target)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        arrays = run_tabular(definitions)
    original=target/'tabular-native.npz'
    if original.exists():
        archive=PROOF/'evidence'/('tabular-native-original-'+sha(original)[:12]+'.npz')
        if not archive.exists():shutil.copy2(original,archive)
        with np.load(original) as retained:
            for key,value in arrays.items():np.testing.assert_array_equal(value,retained[key])
    from tabular_regressions import run_regressions
    with threadpool_limits(limits=1):arrays.update(run_regressions(definitions))
    temporary=target/'tabular-native-next.npz'
    np.savez_compressed(temporary,**arrays)
    os.replace(temporary,original)
    write_json(PROOF / "evidence/tabular-native.json",{key:{"shape":list(value.shape)} for key,value in arrays.items()})


if __name__ == "__main__":
    main()

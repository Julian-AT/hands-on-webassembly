"""Pinned browser arithmetic; estimator, initialization and optimizer stay intact."""
import sys

if sys.platform=='emscripten':
    import sklearn
    if sklearn.__version__!='1.6.1':
        raise RuntimeError('The t-SNE runtime version changed. Reload and retry.')
    import neighbor_compat
    import native_barnes_hut_tsne
    from sklearn.manifold import _t_sne
    _t_sne._barnes_hut_tsne=native_barnes_hut_tsne

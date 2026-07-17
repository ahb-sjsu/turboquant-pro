# Reproduce the canonical embedding_glove_recall claim (full 1.18M GloVe).
# Needs glove-100-angular.hdf5 (ann-benchmarks) + turboquant-pro[yaml].
export TQP_GLOVE_HDF5=/path/to/glove-100-angular.hdf5
tqp replay embedding_glove_recall --full
# equivalently:
python benchmarks/canonical_glove.py --full --out results.json

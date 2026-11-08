export MP_ROOT=$PSCRATCH/masters-project
export GH_ROOT=$HOME/masters-project-github

export PIP_CACHE_DIR=$MP_ROOT/cache/pip
export HF_HOME=$MP_ROOT/cache/huggingface
export XDG_CACHE_HOME=$MP_ROOT/cache/xdg
export NLTK_DATA=$MP_ROOT/cache/nltk
export TMPDIR=$MP_ROOT/tmp
export CONDA_PKGS_DIRS=$MP_ROOT/cache/conda-pkgs

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$MP_ROOT/src/BIPIA:$PYTHONPATH

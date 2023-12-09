# Pairing (initially)
#jupytext --set-formats ipynb,py 01.01_clean_donorinfo.ipynb
#jupytext --set-formats ipynb,py 01.02_clean_group_weights.ipynb
#jupytext --set-formats 01.03_clean_papq.ipynb
#jupytext --set-formats ipynb,py 02.01_merge_cleaned_datasets.ipynb
#jupytext --set-formats ipynb,py 03.02_causal_inference-modelling.ipynb


# Syncing
#jupytext --sync 01.01_clean_donorinfo.ipynb
#jupytext --sync 01.02_clean_group_weights.ipynb
#jupytext --sync 01.03_clean_papq.ipynb
#jupytext --sync 02.01_merge_cleaned_datasets.ipynb
#jupytext --sync 03.01_base_models.ipynb
jupytext --sync 03.02_causal_inference-modelling.ipynb

# Verify and clean
#pre-commit run --all-files

# Sync change back to notebooks
#jupytext --sync 01.01_clean_donorinfo.ipynb
#jupytext --sync 01.02_clean_group_weights.ipynb
#jupytext --sync 01.03_clean_papq.ipynb
#jupytext --sync 02.01_merge_cleaned_datasets.ipynb
#jupytext --sync 03.01_base_models.ipynb

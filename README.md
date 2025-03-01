# FBME Bio Library – Tools for Biomedical and Bioinformatics Research

## Getting started for developing

- download sources
```
git clone https://github.com/kbi-fbmi/biolib.git
```
- vscode: insall recomended extensions .vscode is synced

- requirements in projects.toml use uv for adding packages
```
uv sync --python 3.10
```
- adding packages pyproject.toml (for dev enviroment)
```
uv add <package>
uv add ---dev <package> 
```
- uv installation
```
pipx install uv 
pip install uv
```

## using library

example: https://colab.research.google.com/drive/1ldHDU5FaW6zpttY20-lNbeX0mTTJYbMm
```
pip install git+https://github.com/kbi-fbmi/biolib.git
or
uv add git+https://github.com/kbi-fbmi/biolib.git
```


## comments git store password
linux
```
git config --global credential.helper store
```
windows
```
git config --global credential.helper wincred
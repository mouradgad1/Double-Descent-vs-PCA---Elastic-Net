# Double Descent vs PCA & Elastic Net

Project skeleton and utilities for experimenting with model complexity, dimensionality reduction (PCA),
and regularization (Elastic Net). The goal is to provide reproducible pipelines and small helper
utilities to run experiments that explore double-descent phenomena and how PCA / Elastic Net
affect generalization.

Key ideas

- Compare polynomial regression (baseline) across increasing model complexity.
- Use PCA to control effective dimensionality before fitting a regressor.
- Use Elastic Net (with/without CV) to study the effect of L1/L2 regularization on double-descent.

Repository structure

```
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── base_model.py              # polynomial regression model
│   ├── pca_model.py               # PCA + model 
│   └──elastic_net_model.py       # Elastic Net 
│    
├── figures/                       # (all output plots here)

```

Quick start (PowerShell)

1. Create and activate a virtual environment:

```powershell
python -m venv venv; .\venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the built-in sanity checks for each module (from project root):

```powershell
python src\base_model.py
python src\pca_model.py
python src\elastic_net_model.py
```

If everything is installed correctly, each script should generate a graph in the Figures folder.


Development notes

- `requirements.txt` lists the core Python packages used for reproducible environments.
- `src/` modules include small `__main__` checks so you can quickly verify the pipeline works locally.

Contributing

Suggestions and pull requests welcome.

Contact

For questions or help, open an issue or contact the repository owner.

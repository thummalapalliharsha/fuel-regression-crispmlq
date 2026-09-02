# Flight Fuel-Burn Prediction — Setup & Run Guide (VS Code + MySQL)

Companion to `Supervised_Learning_Algorithms_and_Metrics.html`. Follow the
CRISP-ML(Q) build end to end: load data → MySQL → model → HPO → evaluate → deploy.

## Files
| File | What it is |
|------|------------|
| `flight_fuel_logs.csv` | Raw data — 120 anonymised flights |
| `fuel_regression_crispmlq.py` | Main script, run **cell by cell** in VS Code |
| `app.py` | Streamlit web app for production serving (Phase 5) |
| `fuel_model.joblib` | Saved model (created when you run Phase 5) |

## 1. VS Code setup + virtual environment
1. Install **VS Code**, then the **Python** and **Jupyter** extensions.
2. Open this folder: `File ▸ Open Folder…` → `LinearRegression`.
3. Create an **isolated virtual environment** and install the packages. In the VS Code
   terminal, from inside the `LinearRegression` folder:

   **macOS / Linux**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
   **Windows (PowerShell)**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
   When it is active you'll see `(.venv)` at the start of the terminal prompt.
4. Point VS Code at it: Command Palette (`Ctrl/Cmd+Shift+P`) → **Python: Select
   Interpreter** → choose the one under `.venv`. Now "Run Cell" and the terminal
   both use this environment.

> A `.venv` folder is already created here with everything installed. To start fresh,
> delete `.venv` and repeat step 3. Do **not** commit `.venv` to git (see `.gitignore`).

## 2. MySQL setup
1. Install **MySQL Server** (and MySQL Workbench, optional GUI). Start the service.
2. Create the database once — in MySQL Workbench or the CLI:
   ```sql
   CREATE DATABASE airline;
   ```
   You do **not** need to create the table — `pandas.to_sql` builds it for you.
3. Open `fuel_regression_crispmlq.py` and edit the connection block at the top:
   ```python
   MYSQL_USER = "root"
   MYSQL_PASSWORD = "your_password_here"   # <-- your MySQL password
   MYSQL_HOST = "localhost"
   MYSQL_PORT = 3306
   MYSQL_DB   = "airline"
   ```

## 3. Run the model, one cell at a time
In `fuel_regression_crispmlq.py`, each block starts with `# %%`. Click **Run Cell**
above each block, in order:

| Phase | Cells do… |
|-------|-----------|
| 0 | Imports & config |
| 1 | Load CSV → explore → **push into MySQL** → **pull from MySQL** |
| 2 | Data prep: features/target + preprocessing recipe |
| 3 | Train/test split → OLS **estimate table** → sklearn model → regularization → **HPO (GridSearch + Optuna)** |
| 4 | Evaluation: MAE / MSE / RMSE / MAPE / R² + residual plot |
| 5 | **Deploy**: refit on all data → save `fuel_model.joblib` |
| 6 | Monitoring: simple drift check |

Verify the data landed in MySQL any time:
```sql
SELECT COUNT(*) FROM airline.flight_fuel_logs;   -- 120
SELECT * FROM airline.flight_fuel_logs LIMIT 5;
```

> **MySQL troubleshooting**
> - **Password with special characters** (`@ # : /`) is handled automatically — the
>   script URL-encodes it. (An `@` in the password was previously breaking the URL.)
> - **`Can't connect to MySQL server`** means the server isn't running, the `airline`
>   database doesn't exist, or the password is wrong. Start MySQL, run
>   `CREATE DATABASE airline;`, and set `MYSQL_PASSWORD` at the top of the script.
> - If MySQL still can't be reached, the script **prints a message and falls back to the
>   CSV automatically**, so the rest of the phases still run. Fix MySQL and re-run the two
>   Phase 1 cells to switch back to the database.

## 4. Reading the model estimate output (Phase 3)
`ols.summary()` prints the regression table. Key columns:
- **coef** — the effect: `distance_km ≈ 3.0` → each extra km ≈ +3 kg fuel; `const ≈ 2100` → fixed taxi/take-off/reserve baseline.
- **P>|t|** — p-value; `< 0.05` means the feature genuinely matters (all ours are ~0.000).
- **[0.025, 0.975]** — 95% confidence interval for the coef.
- **R-squared** — share of fuel variation explained (~0.999 here; distance dominates).

Expected test metrics: **MAE ≈ 90 kg, RMSE ≈ 100 kg, MAPE ≈ 1.5%, R² ≈ 0.999**.

## 5. Deploy (Phase 5)
After Phase 5 saves `fuel_model.joblib`, launch the app from the VS Code terminal:
```bash
streamlit run app.py
```
A browser opens; enter a flight's details and get the predicted fuel burn. This is
the production interface dispatchers would use. In a real rollout you would host it
(e.g. a cloud VM / container) and point it at the live MySQL table.

## 6. Push the project to GitHub
The `.venv` folder and `fuel_model.joblib` are excluded by `.gitignore`, so only the
source files get uploaded. A git repo is already initialised here with a first commit.

**a) First-time push — create the GitHub repo, then connect and push**

*Option A — using the GitHub website:*
1. Go to <https://github.com/new>, name it e.g. `linear-regression-fuel`, leave it
   empty (no README/gitignore), click **Create repository**.
2. Copy the repo URL, then in the VS Code terminal (inside `LinearRegression`):
   ```bash
   git remote add origin https://github.com/deepubharani-code/linear-regression-fuel
   git push -u origin main
   ```
   Sign in when prompted (browser login, or a Personal Access Token as the password).

*Option B — using the GitHub CLI (`gh`):*
```bash
gh auth login                       # one-time sign-in
gh repo create linear-regression-fuel --private --source=. --remote=origin --push
```

**b) Push changes later — the everyday loop**

Whenever you edit the code, repeat these three steps:
```bash
git add .                                   # stage all changes
git commit -m "Describe what you changed"   # save a snapshot
git push                                    # upload to GitHub
```
Check what changed before committing with `git status` and `git diff`. In VS Code you
can also use the **Source Control** panel (left sidebar) to stage, commit and push
with buttons instead of commands.

> If you named the repo differently or already have a remote, update it with:
> `git remote set-url origin <new-url>`. See the current remote with `git remote -v`.

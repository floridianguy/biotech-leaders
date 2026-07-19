# Biotech Leaders

This project downloads historical stock prices, ranks biotech stocks using three trading systems, and publishes a static dashboard to GitHub Pages.

## How the hosted dashboard works

The workflow in `.github/workflows/deploy-pages.yml` runs automatically at 23:17 UTC every weekday and can also be started manually. It:

1. Discovers a biotech ticker universe, falling back to `biotech_universe.csv` if discovery is unavailable.
2. Generates `output/biotech_rankings.xlsx`.
3. Converts the dashboard sheets to browser-friendly JSON.
4. Deploys the static site and downloadable workbook as a GitHub Pages artifact.

Generated rankings are not committed to repository history.

## Enable GitHub Pages

After pushing this project to a GitHub repository:

1. Open the repository on GitHub.
2. Select **Settings**, then **Pages**.
3. Under **Build and deployment**, set **Source** to **GitHub Actions**.
4. Open **Actions**, select **Refresh rankings and deploy GitHub Pages**, and choose **Run workflow** for the first deployment.
5. After the workflow finishes, its deployment job displays the public Pages URL.

The workflow expects the repository's default branch to be named `main`. If yours has a different name, update the `push.branches` value in the workflow. Scheduled workflows always execute from the default branch.

## Local setup

Create a virtual environment and install the dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Generate fresh rankings:

```powershell
python -m src.biotech_ranker --discover --tickers biotech_universe.csv --output output/biotech_rankings.xlsx
```

Build the static site from an existing workbook:

```powershell
python build_static_site.py
```

Preview it locally (opening `_site/index.html` directly will not load its JSON in most browsers):

```powershell
python -m http.server 8000 --directory _site
```

Then visit `http://localhost:8000`.

## Tests

```powershell
python -m pytest -q
```

The dashboard is public and intended for informational purposes only. Confirm that your market-data provider permits redistribution before sharing the site broadly.

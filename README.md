# Two-stage landslide localization

Run all commands from the repository root. Place the requested files as shown in
[data placement](docs/data.md).

## Reproduce the experiments and tables

Use Python 3.12 with compatible GDAL/PROJ libraries. Install dependencies and set
the geographic-data paths in `configs/paths.local.json`:

```bash
python -m pip install -r requirements.txt
cp configs/paths.example.json configs/paths.local.json
```

Run Full and all five ablations, then Direct LLM, and collect their tables:

```bash
python scripts/run_full.py
python scripts/run_direct.py
python scripts/collect_run_metrics.py
```

Results are written to `outputs/`. To regenerate Direct LLM answers, follow the
[inference instructions](docs/direct_inference.txt).

## Data availability

Request data from *bing.xu@uni-jena.de* for a reasonable research
purpose, subject to [X's content redistribution policy](https://docs.x.com/developer-terms/policy#content-redistribution)
and applicable third-party terms. See [data terms](docs/DATA_LICENSE.md).

## External model

[Mistral-7B finetuned with LoRA](https://github.com/xukehu/LLM-geocoding)

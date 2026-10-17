# Data placement

## Collect the experiment tables

Place the supplied annotations and test partition at:

```text
data/public/annotations.json
data/public/splits/test.json
```

After running Full and Direct LLM, `python scripts/collect_run_metrics.py`
uses these files to summarize the recomputed results. Use `--annotations` and
`--test-split` to read copies stored elsewhere.

## Spatial experiments

Place the supplied full records and process inputs at:

```text
data/full/
  annotations.json
  splits/
    development.json
    test.json
    event_groups.json
    split_summary.json
  process/
    candidates.json
    city_boundary_overrides.json
    image_semantics.json
    direct_answers.json
    direct_answer_cache/*.json
```

Use the complete supplied geocoder cache, including empty responses. Full
annotations contain the fields needed by spatial evaluation; the reduced
`data/public/annotations.json` is used to collect the experiment tables.

Copy `configs/paths.example.json` to `configs/paths.local.json` and set the
following geographic inputs. Paths may be absolute or relative to the repository
root; supplied files may also be kept outside the repository.

| Configuration key | Input |
|---|---|
| `susceptibility_raster` | Experiment susceptibility GeoTIFF (`n10_conus.tif`) |
| `rainfall_dir` | PRISM daily precipitation rasters in year directories |
| `osm_state_dir` | State OSM `.osm.pbf` extracts |
| `road_state_dir` | Road-only state `.osm.pbf` extracts |
| `hydro_state_dir` | Water-only state `.osm.pbf` extracts |
| `national_road_pbf` | National road PBF fallback |

Use the original spatial source versions. PRISM paths follow
`2020/PRISM_ppt_stable_800mD2_20200101_bil.bil`, with companion headers and
projection files, covering the 1-, 3-, 5-, and 7-day rainfall windows. State PBF
names follow `Alabama.osm.pbf` and `New_York.osm.pbf`.

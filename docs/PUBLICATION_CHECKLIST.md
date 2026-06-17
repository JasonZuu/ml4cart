# Code Publication Checklist

This checklist maps the requirements from `GuidelinesCodePublication.pdf` to
files in this repository.

## Submission Requirements

| Requirement | Repository location | Status |
| --- | --- | --- |
| Source code with version details | `pyproject.toml`, `VERSION`, module directories | Complete |
| README listing provided documentation | `README.md` | Complete |
| Installation guide with OS, language, dependencies, hardware/resources, install time | `docs/INSTALLATION.md`, `requirements.txt` | Complete |
| Demo on example data with typical runtime | `docs/DEMO.md`, `*/demo_data/` | Complete |
| Test/example dataset description | `docs/TEST_DATA.md` | Complete |
| Key operations, tasks, approach, and characteristics | `docs/METHODS_OVERVIEW.md` | Complete |
| Parameters and outputs | `docs/PARAMETERS.md` | Complete |
| License of use | `LICENSE`, `docs/CODE_AVAILABILITY.md` | Complete |
| Open repository link and DOI, when available | `docs/CODE_AVAILABILITY.md` | Pending final repository/DOI assignment |
| Community commenting/input link | `docs/CODE_AVAILABILITY.md` | Pending final public repository setup |

## Acceptance Requirements

| Requirement | Repository location | Status |
| --- | --- | --- |
| Paper-associated software version | `VERSION`, release tag `v0.1.0` | Ready for release |
| Associated test data | `*/demo_data/` | Complete |
| Parameters and documentation | `docs/`, folder READMEs | Complete |
| Maintained repository or archived zip | `docs/CODE_AVAILABILITY.md` | Pending final archive |
| Code availability statement | `docs/CODE_AVAILABILITY.md` | Template provided |

## Release Packaging

Recommended release archive command:

```bash
cd /home/jasonz/Code/ml4cart-private
zip -r ML4CAR-T-v0.1.0.zip ml4cart \
  -x 'ml4cart/.git/*' \
  -x 'ml4cart/**/__pycache__/*' \
  -x 'ml4cart/**/.pytest_cache/*' \
  -x 'ml4cart/**/results/*' \
  -x 'ml4cart/**/wandb/*' \
  -x 'ml4cart/GuidelinesCodePublication.pdf'
```

The publication archive should include code, `docs/`, tests, and `demo_data/`,
but should exclude generated results, caches, private data, and the guideline
PDF itself.

"""Runs `al-analysis` (`al_analysis.al_analysis.main`) for `test_characterize_al.py`.

Run in a subprocess (one per `al-analysis` run), since `main` keeps state in module
globals (lists of per-recording outputs, the driver/indicator combination, ...), so
two runs in one process would not be independent.

Usage (the environment must already point hong2p at the test data tree, via
HONG2P_DATA / HONG2P_FAST_DATA, and XDG_DATA_HOME at a temp dir, so hong2p's odor
abbreviation cache is not read from / written to the user's):
```
python test/al_cli_driver.py <gsheet-json> [al-analysis CLI args...]
```
`<gsheet-json>` is a JSON list of dicts with the metadata Google Sheet columns for each
fly (date, fly, driver, indicator, side, ...), used in place of the Google Sheet (which
needs network access and a link file).

The current directory should be the output directory (`al-analysis` writes outputs
under `<driver>_<indicator>/` relative to it).

Patches applied, each to work around something that would otherwise stop the run (in the
current venv, with Python 3.14 and xarray 2026, or on a host without some inputs), and
none of which change outputs of a working run:
- the multiprocessing start method is set to 'fork' (the default before Python 3.14),
  because `al_analysis.logging.init_logger` installs `multiprocessing_logging`'s handler,
  which asserts the 'fork' start method (Python 3.14 defaults to 'forkserver' on Linux).

- `rois2best_planes_only` (from hong2p.roi) is wrapped to restore its old-xarray
  output: it ends with `rois.assign_coords(roi=rois.roi_name)`, which newer xarray
  ignores when 'roi' is a MultiIndex (emitting a FutureWarning, which mb_model's
  module-level filter turns into an error), so best-plane ROIs keep the full MultiIndex
  instead of the ROI names, and `ij_traces` then fails an assertion.

- `do_register_all_fly_recordings_together` and `do_convert_raw_to_tiff` are set False
  (module-level switches), so suite2p registration and the raw -> TIFF conversion are
  not run. The test provides motion corrected TIFFs from the NAS instead.

- if AL_CLI_DRIVER_OLF_CONFIG_DIR is set, olfactometer input YAMLs (for the odor
  metadata output) are read from there, rather than from ~/src/tom_olfactometer_configs.

If AL_ANALYSIS_DATA_AUDIT is set, the data audit from `test/conftest.py` is installed
here too, with the test ID from `AL_ANALYSIS_DATA_AUDIT_TEST`.
"""

import json
import multiprocessing
import os
from pathlib import Path
import sys
import warnings


def main():
    gsheet_json = Path(sys.argv[1])
    cli_args = sys.argv[2:]

    if os.environ.get('AL_ANALYSIS_DATA_AUDIT'):
        # (this file's directory is sys.path[0] when run as a script)
        import conftest
        conftest.pytest_configure(None)
        conftest._audit_state['current'] = os.environ.get(
            'AL_ANALYSIS_DATA_AUDIT_TEST', '<al_cli_driver>'
        )

    multiprocessing.set_start_method('fork', force=True)

    import matplotlib
    matplotlib.use('agg')

    import pandas as pd

    import al_analysis.al_analysis as ala
    from al_analysis import al_util

    rows = json.loads(gsheet_json.read_text())

    def get_gsheet_metadata() -> pd.DataFrame:
        df = pd.DataFrame(rows)
        df['date'] = pd.to_datetime(df['date'])
        return df.set_index(['date', 'fly'], verify_integrity=True)

    ala.get_gsheet_metadata = get_gsheet_metadata
    al_util.get_gsheet_metadata = get_gsheet_metadata

    orig_rois2best_planes_only = ala.rois2best_planes_only

    def rois2best_planes_only(*args, **kwargs):
        roi_indices, rois = orig_rois2best_planes_only(*args, **kwargs)
        index = rois.indexes.get('roi')
        if isinstance(index, pd.MultiIndex):
            names = rois.roi_name.values
            rois = rois.drop_vars(['roi'] + list(index.names)).assign_coords(
                roi=('roi', names)
            )
        return roi_indices, rois

    ala.rois2best_planes_only = rois2best_planes_only

    ala.do_register_all_fly_recordings_together = False
    ala.do_convert_raw_to_tiff = False

    olf_config_dir = os.environ.get('AL_CLI_DRIVER_OLF_CONFIG_DIR')
    if olf_config_dir is not None:
        orig_load_olf_input_yaml = ala.load_olf_input_yaml

        def load_olf_input_yaml(yaml_name, olf_config_dir=Path(olf_config_dir),
            **kwargs):
            return orig_load_olf_input_yaml(yaml_name, olf_config_dir=olf_config_dir,
                **kwargs
            )

        ala.load_olf_input_yaml = load_olf_input_yaml

    sys.argv = ['al-analysis'] + cli_args
    with warnings.catch_warnings():
        # (in front of mb_model's filter making all FutureWarnings errors)
        warnings.filterwarnings('ignore', category=FutureWarning,
            message="updating coordinate 'roi', which is a PandasMultiIndex"
        )
        ala.main()


if __name__ == '__main__':
    main()

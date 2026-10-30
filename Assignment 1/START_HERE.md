# MMUQ Assignment 01

## Read the result

Open `calibration_outputs/assignment_report.md`. It explains the objective, model
evaluation, optimizer settings, fitted parameters, convergence, all required plots,
and the limits of the result.

## Read the code

1. `calibration_config.py`: the supplied bounds and explicit settings.
2. `ab_calibrate_hbv.py`: start at `main()`, then follow the named functions.
3. `calibration_report.py`: formats the result report from actual saved values.
4. `ac_verify_results.py`: checks saved records and replays candidate simulations.

`hmg/` and its original demonstration are the supplied model, unchanged.
`references/` contains the two supplied PDFs and the parameter-bound screenshot.
The earlier 100-generation backup is stored outside this upload folder.
`calibration_outputs/` contains the final converged result.

## Run

After installing `requirements.txt` into a Python 3.12 environment, macOS users can run:

```sh
./run_calibration.command --check
./run_calibration.command
```

On another machine with Python 3.12, install the listed dependencies and run from
this directory:

```sh
python -m pip install -r requirements.txt
python ab_calibrate_hbv.py --check
python ab_calibrate_hbv.py
python ac_verify_results.py
```

The full calibration overwrites `calibration_outputs/`. The paths are relative to
the script location, so the data and model folders must remain beside the scripts.
The Mac launcher uses `python3` by default. Set `PYTHON_EXECUTABLE` to select another
Python environment where the dependencies have been installed. No personal machine
paths or separately stored dependencies are required by the project code.

## Scope

All 18 supplied parameter entries are calibrated on the complete provided hourly
record, using `1 - NSE`. Initial storages are included in the vector. No warm-up
is excluded and no independent validation period is claimed. These choices and all
optimizer settings are declared explicitly; the written assignment does not specify
a different split. Any additional classroom or Moodle submission rules must still
be checked against the course page.

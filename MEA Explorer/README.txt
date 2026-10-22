MEA Analyzer  --  standalone home for the MEA-NAP tool
======================================================

FOLDER LAYOUT (everything the tool needs lives HERE)
    MEA Analyzer\
        MEA Analyzer.bat      <- double-click to launch the GUI
        README.txt
        Scripts\MEA-NAP\      <- the pipeline code (move here)
        .venv\                <- the Python environment (move here)

The experiment DATA stays in its own folder (H5\, Output\, ...). You point
the GUI's Data / Spike / Plot fields at whichever experiment you want, so this
one tool serves every experiment. Use "Save config..." to keep a preset per
experiment.

TO FINISH SETTING THIS UP (one-time)
    1. In Explorer, CUT these two folders from the old location
           ...\New MEA 06172026\Scripts
           ...\New MEA 06172026\.venv
       and PASTE them into THIS folder, so you end up with:
           MEA Analyzer\Scripts\MEA-NAP\...
           MEA Analyzer\.venv\Scripts\python.exe
    2. Double-click "MEA Analyzer.bat".  If the GUI opens, you're done.
    3. In the GUI, set the Data / Spike output / Plot output folders to your
       experiment, then Save config... so it's one click next time.

IF THE GUI DOES NOT OPEN AFTER MOVING .venv
    Moving a virtual environment usually works (the launcher calls
    python.exe directly), but if it doesn't, either:
      - set MEANAP_PYTHON in "MEA Analyzer.bat" to your python.exe, OR
      - recreate the venv here:  python -m venv .venv   then reinstall the
        dependencies (note: the CUDA build of torch needs its own install
        index; a plain "pip install torch" gives the CPU build).

NOTES
    - You can delete any old "__pycache__" folders; they regenerate.
    - Do NOT let a cloud sync copy the multi-GB .venv around; exclude it
      from sync if this folder is inside a synced location.
    - Leftover experiment data in the old Scripts folder (e.g. a stray .h5
      or the .seq pattern file) can stay with the experiment; the tool does
      not need them at runtime.

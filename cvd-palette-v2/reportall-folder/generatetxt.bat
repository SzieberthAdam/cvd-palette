@echo off
setlocal enabledelayedexpansion
for %%f in (*.png) do (
echo Processing %%f ...
    set "filename=%%~nf"
python ..\reportall.py "%%f
echo Done.
)

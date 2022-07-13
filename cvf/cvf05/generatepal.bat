@echo off
setlocal enabledelayedexpansion
for %%f in (*.png) do (
echo Processing %%f ...
    set "filename=%%~nf"
python ..\..\png2pal.py "%%f
echo Done.
)

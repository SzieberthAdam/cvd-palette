@echo off
setlocal enabledelayedexpansion
for %%f in (*.01x.png) do (
echo Processing %%f ...
    set "filename=%%~nf"
python ..\..\png2pal.py "%%f
echo Done.
)

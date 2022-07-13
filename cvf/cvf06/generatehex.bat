@echo off
setlocal enabledelayedexpansion
for %%f in (*.pal) do (
echo Processing %%f ...
    set "filename=%%~nf"
python ..\..\pal2hex.py "%%f
echo Done.
)

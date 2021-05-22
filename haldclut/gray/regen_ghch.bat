@echo off
for %%f in (*.png) do (
echo %%f
python png2ghch.py "%%f"
)
pause


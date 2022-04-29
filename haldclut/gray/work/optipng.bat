@echo off
for %%f in (*.png) do (
echo %%f
optipng -o7 -nx "%%f"
)
pause

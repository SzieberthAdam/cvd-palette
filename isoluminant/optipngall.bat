@echo off
echo Optimizing PNG files...
for %%f in (level%1\*.png) do (
echo %%f
optipng -o7 -nx -quiet "%%f"
)
echo Done.

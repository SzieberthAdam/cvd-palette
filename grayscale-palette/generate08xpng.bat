@echo off
setlocal enabledelayedexpansion
for %%f in (*.01x.png) do (
echo Processing %%f ...
    set "filename=%%~nf"
magick convert "%%f" -interpolate Nearest -filter point -resize 800%% png24:"!filename:~0,-4!.08x.png"
echo Done.
)


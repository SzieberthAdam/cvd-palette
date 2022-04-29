@echo off
for %%f in (*.png) do (
echo %%f
magick convert %%f BMP2:%%~nf.bmp
)
pause

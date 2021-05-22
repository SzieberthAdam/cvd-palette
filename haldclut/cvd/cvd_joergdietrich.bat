@echo off
echo https://github.com/joergdietrich/daltonize
echo.
echo "detua.joergdietrich.png" ...
daltonize -s -t d "%0\..\..\identity\identity.png" "detua.joergdietrich.png"
echo done.
echo "prota.joergdietrich.png" ...
daltonize -s -t p "%0\..\..\identity\identity.png" "prota.joergdietrich.png"
echo done.
echo "trita.joergdietrich.png" ...
daltonize -s -t t "%0\..\..\identity\identity.png" "trita.joergdietrich.png"
echo done.
pause

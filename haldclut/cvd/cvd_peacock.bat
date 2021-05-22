@echo off
echo https://github.com/jkulesza/peacock
echo.
echo "achra.peacock.png" ...
python "%0\..\peacock\peacock.py" --cb Monochromacy "%0\..\..\identity\identity.png"
move "%0\..\..\identity\identity_Monochromacy.png" "achra.peacock.png"
echo done.
echo "deuta.peacock.png" ...
python "%0\..\peacock\peacock.py" --cb Deuteranopia "%0\..\..\identity\identity.png"
move "%0\..\..\identity\identity_Deuteranopia.png" "deuta.peacock.png"
echo done.
echo "prota.peacock.png" ...
python "%0\..\peacock\peacock.py" --cb Protanopia "%0\..\..\identity\identity.png"
move "%0\..\..\identity\identity_Protanopia.png" "prota.peacock.png"
echo done.
echo "trita.peacock.png" ...
python "%0\..\peacock\peacock.py" --cb Tritanopia "%0\..\..\identity\identity.png"
move "%0\..\..\identity\identity_Tritanopia.png" "trita.peacock.png"
echo done.
echo "deuty.peacock.png" ...
python "%0\..\peacock\peacock.py" --cb Deuteranomaly "%0\..\..\identity\identity.png"
move "%0\..\..\identity\identity_Deuteranomaly.png" "deuty.peacock.png"
echo done.
echo "proty.peacock.png" ...
python "%0\..\peacock\peacock.py" --cb Protanomaly "%0\..\..\identity\identity.png"
move "%0\..\..\identity\identity_Protanomaly.png" "proty.peacock.png"
echo done.
echo "trity.peacock.png" ...
python "%0\..\peacock\peacock.py" --cb Tritanomaly "%0\..\..\identity\identity.png"
move "%0\..\..\identity\identity_Tritanomaly.png" "trity.peacock.png"
echo done.
pause

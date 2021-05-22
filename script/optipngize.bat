@echo off
setlocal

IF "%1"=="" GOTO HAVE_0

:HAVE_SOME
for %%f in (%*) do (
echo.
optipng -o7 -nx "%%f"
echo.
)

GOTO END

:HAVE_0
for %%f in (*.png) do (
echo.
optipng -o7 -nx "%%f"
echo.
)
GOTO END

:END
endlocal


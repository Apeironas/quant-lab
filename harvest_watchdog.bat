@echo off
REM =====================================================================
REM  HARVEST BEKÇİSİ - funding-harvest ileri doğrulama takipçisi.
REM  Günde bir canlı evreni tarar, sepeti (7 günde bir) yeniler, simüle
REM  geliri data\harvest\harvest_log.csv'ye yazar. EMİR GÖNDERMEZ (paper).
REM  Kullanım: çift tıkla, pencereyi açık bırak (3. kalıcı pencere).
REM =====================================================================
cd /d "%~dp0"
:loop
echo [%date% %time%] Harvest takipcisi baslatiliyor...
.venv\Scripts\python.exe -m src.tools.harvest_tracker --loop
echo [%date% %time%] Harvest kapandi (kod: %errorlevel%). 60 sn sonra tekrar...
timeout /t 60 /nobreak >nul
goto loop

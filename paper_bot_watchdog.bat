@echo off
REM =====================================================================
REM  PAPER BOT BEKÇİSİ (watchdog)
REM  ML stratejili paper-trading botu ne sebeple kapanırsa kapansın
REM  15 sn sonra yeniden başlatır. Emirler ASLA borsaya gitmez (mode: paper).
REM  Kullanım: çift tıkla, pencereyi açık bırak. Durdurmak: pencereyi kapat.
REM  (Collector bekçisinden AYRI bir penceredir - ikisi birlikte çalışır.)
REM =====================================================================
cd /d "%~dp0"
:loop
echo [%date% %time%] Paper bot baslatiliyor...
.venv\Scripts\python.exe -m src.main
echo [%date% %time%] Paper bot kapandi (kod: %errorlevel%). 15 sn sonra tekrar...
timeout /t 15 /nobreak >nul
goto loop

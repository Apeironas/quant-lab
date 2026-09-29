@echo off
REM =====================================================================
REM  L2 TOPLAYICI BEKÇİSİ (watchdog)
REM  Toplayıcı ne sebeple kapanırsa kapansın 10 sn sonra yeniden başlatır.
REM  Kullanım: bu dosyaya çift tıkla (veya Görev Zamanlayıcı'ya ekle).
REM  Durdurmak için: pencereyi kapat.
REM =====================================================================
cd /d "%~dp0"
:loop
echo [%date% %time%] Toplayici baslatiliyor...
.venv\Scripts\python.exe -m src.data.orderbook_collector
echo [%date% %time%] Toplayici kapandi (kod: %errorlevel%). 10 sn sonra tekrar...
timeout /t 10 /nobreak >nul
goto loop

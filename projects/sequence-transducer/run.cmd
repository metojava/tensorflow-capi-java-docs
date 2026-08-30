@echo off
rem Task-conditioned seq2seq transducer: LSTM encoder + dense decoder (reverse/sort).
rem Usage: run.cmd [epochs]
setlocal
set "TF_CAPI_NATIVE_DIR=C:\Users\dell\Desktop\libtensorflow-cpu-windows-x86_64\lib"
set "MAVEN_OPTS=-Xmx1g"
cd /d "%~dp0"
mvn -q compile exec:java %*
endlocal

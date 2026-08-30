@echo off
rem MNIST digit CNN: train Conv2D+MaxPool net on handwritten digits.
rem Usage: run.cmd [datadir] [epochs] [batch]
setlocal
set "TF_CAPI_NATIVE_DIR=C:\Users\dell\Desktop\libtensorflow-cpu-windows-x86_64\lib"
set "MAVEN_OPTS=-Xmx1g"
cd /d "%~dp0"
mvn -q compile exec:java %*
endlocal

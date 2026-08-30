@echo off
rem Churn prediction: one-hot encoded telco customers, stays vs churns.
rem Usage: run.cmd
setlocal
set "TF_CAPI_NATIVE_DIR=C:\Users\dell\Desktop\libtensorflow-cpu-windows-x86_64\lib"
set "MAVEN_OPTS=-Xmx1g"
cd /d "%~dp0"
mvn -q compile exec:java %*
endlocal

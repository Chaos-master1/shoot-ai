Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "C:\Users\Expert Gaming\Downloads\football coach ai"
WshShell.Environment("PROCESS")("PYTHONPATH") = "src"
WshShell.Run "cmd /c ""C:\Program Files\Python311\python.exe"" scripts\download_tracking_only.py > data\soccernet\stdout_tracking.log 2>&1", 0, False

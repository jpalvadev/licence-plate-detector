@REM  modo debug en otra ventana
@REM   C:\Programacion\mediamtx && ./mediamtx.exe 
@REM  ffmpeg -re -stream_loop -1 -i sample.mp4 -map 0:v -c copy -f rtsp rtsp://127.0.0.1:8554/mystream

@REM  si es modo debug correr antes los comandos anteriores y luego
.venv/Scripts/python.exe -m alpr
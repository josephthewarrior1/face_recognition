@echo off
echo ============================================
echo BUILDING FACE ATTENDANCE SYSTEM .EXE
echo Firebase Edition
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed!
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

echo [1/4] Installing dependencies...
echo.
pip install -r requirements.txt

echo.
echo [2/4] Creating .exe with PyInstaller...
echo.

REM Create executable
pyinstaller --onefile ^
    --windowed ^
    --name="FaceAttendance_Firebase" ^
    --icon=NONE ^
    --add-data "detector.tflite;." ^
    --hidden-import=cv2 ^
    --hidden-import=mediapipe ^
    --hidden-import=firebase_admin ^
    --hidden-import=tkinter ^
    face_attendance_firebase.py

echo.
echo [3/4] Checking if build was successful...
if exist "dist\FaceAttendance_Firebase.exe" (
    echo.
    echo [4/4] SUCCESS! Executable created!
    echo.
    echo ============================================
    echo FILE LOCATION:
    echo %CD%\dist\FaceAttendance_Firebase.exe
    echo ============================================
    echo.
    echo IMPORTANT NOTES:
    echo 1. Copy firebase-config.json to the same folder as the .exe
    echo 2. Download detector.tflite model (will auto-download on first run)
    echo 3. The .exe can be distributed to any Windows PC
    echo 4. All computers using the .exe will share the same Firebase database
    echo.
    echo ============================================
    
    REM Open dist folder
    echo Opening dist folder...
    explorer dist
    
) else (
    echo.
    echo ERROR: Build failed!
    echo Check the error messages above.
)

echo.
pause
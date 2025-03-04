pyinstaller --name="Contour Plot for 9A" ^
            --icon=9A.ico ^
            --windowed ^
            --onefile ^
            --clean ^
            --noupx ^
            --add-data="9A.ico;." ^
            --hidden-import=numpy ^
            --hidden-import=pandas ^
            --hidden-import=matplotlib ^
            --hidden-import=scipy ^
            --hidden-import=pyqtgraph ^
            --hidden-import="pyimod02_importers" ^
            --hidden-import=openpyxl.cell._writer ^
            --exclude-module=tkinter ^
            --exclude-module=notebook ^
            --exclude-module=IPython ^
            --exclude-module=pytest ^
            contour_main.py


"C:\Program Files (x86)\Windows Kits\10\bin\10.0.22000.0\x64\signtool.exe" sign /f mycert.pfx /p "kgZ2307097%" /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 "C:\Users\user\OneDrive - postech.ac.kr\Python\SAXS_InSitu_Contour_Corrector\dist\Contour Plot for 9A.exe"
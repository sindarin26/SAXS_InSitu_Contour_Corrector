import os
import subprocess
import sys

def convert_ui_to_py(ui_dir):
    """주어진 디렉토리 내 모든 .ui 파일을 .py 파일로 변환하고 UI 스타일을 개선"""
    if not os.path.exists(ui_dir):
        print(f"디렉토리 '{ui_dir}'가 존재하지 않습니다.")
        return

    # if os.name == "nt":
    #     pyuic5_path = r"C:\Users\YoungHyunKim\AppData\Local\anaconda3\envs\FTIR\Scripts\pyuic5.exe"
    # else:
    #     pyuic5_path = "pyuic5"
    pyuic5_path = "pyuic5"

    output_dir = os.path.join(ui_dir, "asset")
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(ui_dir):
        if file.endswith(".ui"):
            ui_path = os.path.join(ui_dir, file)
            py_path = os.path.join(output_dir, file.replace(".ui", ".py"))
            command = f'"{pyuic5_path}" "{ui_path}" -o "{py_path}"'
            try:
                subprocess.run(command, shell=True, check=True)
                print(f"변환 완료: {py_path}")

                # 변환된 .py 파일을 수정하여 UI 스타일 관련 설정 추가
                with open(py_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                insert_index = -1
                for i, line in enumerate(lines):
                    if line.strip().startswith("from PyQt5.QtWidgets import"):
                        insert_index = i + 1  # import 다음 줄에 추가
                        break  

                # UI 스타일 개선 코드 (Fusion + 다크 테마 + 폰트 + 스타일시트 적용)
                ui_style_code = '''
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont, QPalette, QColor

QApplication.setStyle("Fusion")

app = QApplication.instance()
if app:
    # 폰트 설정
    font = QFont("Segoe UI", 10)  # 윈도우에서 가독성이 좋은 폰트
    app.setFont(font)

    # 다크 모드 팔레트 설정
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))  # 다크 배경
    palette.setColor(QPalette.WindowText, QColor(255, 255, 255))  # 흰색 글자
    palette.setColor(QPalette.Base, QColor(35, 35, 35))  # 입력창 배경
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
    palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
    palette.setColor(QPalette.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Highlight, QColor(142, 45, 197))  # 강조 색상
    palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)

    # 기본 스타일시트 적용 (버튼, 입력창 등)
    app.setStyleSheet("""
        QWidget {
            background-color: #353535;
            color: #ffffff;
        }
        QPushButton {
            background-color: #2c3e50;
            border: 1px solid #1a252f;
            padding: 5px;
            border-radius: 3px;
        }
        QPushButton:hover {
            background-color: #1a252f;
        }
        QLineEdit {
            background-color: #1e1e1e;
            border: 1px solid #5a5a5a;
            padding: 4px;
            color: white;
        }
        QLabel {
            font-size: 12px;
            font-weight: bold;
        }
    """)
'''

                if insert_index != -1:
                    lines.insert(insert_index, ui_style_code)
                else:
                    lines.insert(0, "from PyQt5.QtWidgets import QApplication\n")
                    lines.insert(1, ui_style_code)

                with open(py_path, "w", encoding="utf-8") as f:
                    f.writelines(lines)

                print(f"Fusion 스타일 및 UI 개선 적용 완료: {py_path}")

            except subprocess.CalledProcessError as e:
                print(f"오류 발생: {e}")
                sys.exit(1)

if __name__ == "__main__":
    ui_file_dir = r"/Users/yhk/Library/CloudStorage/OneDrive-postech.ac.kr/Python/SAXS_InSitu_Contour_Corrector"
    if ui_file_dir:
        convert_ui_to_py(ui_file_dir)
    else:
        print("ui_file_dir을 설정하세요.")

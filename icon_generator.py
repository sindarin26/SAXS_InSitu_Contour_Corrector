from PIL import Image
import os
from pathlib import Path

def make_square(image: Image.Image) -> Image.Image:
    """원본 이미지를 정사각형으로 맞추기 위해 투명한 배경으로 패딩"""
    width, height = image.size
    max_dim = max(width, height)

    new_image = Image.new("RGBA", (max_dim, max_dim), (0, 0, 0, 0))
    new_image.paste(image, ((max_dim - width) // 2, (max_dim - height) // 2))

    return new_image

def save_icon_variants(image_path: str):
    """PNG를 정사각형으로 변환하고 다양한 아이콘 포맷으로 저장"""
    image_path = Path(image_path)
    if not image_path.exists():
        print("파일을 찾을 수 없습니다:", image_path)
        return
    
    # 저장할 디렉터리 설정
    icon_dir = image_path.parent / "icon"
    icon_dir.mkdir(exist_ok=True)

    # 이미지 열기
    img = Image.open(image_path).convert("RGBA")
    
    # 정사각형으로 변환
    square_img = make_square(img)

    # 아이콘 파일명 설정
    base_name = image_path.stem
    ico_path = icon_dir / f"{base_name}.ico"
    png_path = icon_dir / f"{base_name}.png"
    icns_path = icon_dir / f"{base_name}.icns"

    # 다양한 크기의 아이콘 생성
    icon_sizes = [16, 24, 32, 48, 64, 128, 256, 512, 1024]  # Windows & macOS 지원 크기
    icons = [square_img.resize((size, size), Image.LANCZOS) for size in icon_sizes]

    # PNG 저장
    square_img.save(png_path, format="PNG")

    # ICO 저장 (Windows용)
    icons[0].save(ico_path, format="ICO", sizes=[(size, size) for size in icon_sizes])

    # ICNS 저장 (macOS용)
    if square_img.size[0] >= 1024:
        icns_img = square_img.resize((1024, 1024), Image.LANCZOS)
        icns_img.save(icns_path, format="ICNS")

    print(f"아이콘 저장 완료: {icon_dir}")



# 사용 예시
save_icon_variants(r"C:\Users\user\Pictures\9A.png")

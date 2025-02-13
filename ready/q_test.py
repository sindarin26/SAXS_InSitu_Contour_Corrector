import numpy as np

def q_to_2theta(q, energy_keV):
    """
    q 값을 2θ 값(degree)으로 변환하는 함수.
    
    Parameters:
        q (float or array): q 값 (Å^-1)
        energy_keV (float): X-ray 에너지 (keV)
    
    Returns:
        float or array: 변환된 2θ 값 (degree)
    """
    wavelength = 12.398 / energy_keV  # λ = 12.398 / E
    theta_rad = np.arcsin((q * wavelength) / (4 * np.pi))  # 라디안 단위
    theta_deg = np.degrees(2 * theta_rad)  # 도(degree) 단위 변환
    return theta_deg

def theta_to_q(theta_2, energy_keV):
    """
    2θ 값을 q 값(Å^-1)으로 변환하는 함수.
    
    Parameters:
        theta_2 (float or array): 2θ 값 (degree)
        energy_keV (float): X-ray 에너지 (keV)
    
    Returns:
        float or array: 변환된 q 값 (Å^-1)
    """
    wavelength = 12.398 / energy_keV  # λ = 12.398 / E
    theta_rad = np.radians(theta_2 / 2)  # θ = 2θ / 2
    q_value = (4 * np.pi / wavelength) * np.sin(theta_rad)
    return q_value

def max_SDD_calculation(theta_2_max, pixel_size, beam_center_x, beam_center_y, image_size):
    """
    최대 SDD 값을 계산하는 함수 (MATLAB 좌표 기준).
    
    Parameters:
        theta_2_max (float): 최대 2θ 값 (degree)
        pixel_size (float): 픽셀 크기 (mm)
        beam_center_x (float): 빔 센터 x 좌표 (MATLAB 기준)
        beam_center_y (float): 빔 센터 y 좌표 (MATLAB 기준)
        image_size (tuple): 이미지 크기 (width, height)
    
    Returns:
        tuple: (최대 SDD 값 (mm), 최대 R 값 (mm))
    """
    theta_max_rad = np.radians(theta_2_max)
    corners = [(1, 1), (1, image_size[1]), (image_size[0], 1), (image_size[0], image_size[1])]
    
    R_pixel_max = max(np.sqrt((corner_x - beam_center_x) ** 2 + 
                             (corner_y - beam_center_y) ** 2) 
                      for corner_x, corner_y in corners)
    corrected_R_pixel_max = round(R_pixel_max, 0) - 1
    R_mm_max = corrected_R_pixel_max * pixel_size
    
    SDD_max = R_mm_max / np.tan(theta_max_rad)
    return SDD_max, R_mm_max

def calculate_corrected_sdd(original_2theta, original_sdd, corrected_2theta, exp_energy, converted_energy=8.042):
    if converted_energy == None:
        q_original = original_2theta
        q_corrected = corrected_2theta
    else:
        # 원본 Cu Kα 2θ에 해당하는 q 값 계산
        q_original = theta_to_q(original_2theta, converted_energy)
        q_corrected = theta_to_q(corrected_2theta, converted_energy)
    
    # 이 q 값을 실험 에너지에서의 2θ로 변환
    exp_2theta = q_to_2theta(q_original, exp_energy)
    
    # 이미지상의 R 계산 (실험 에너지 기준)
    original_r = original_sdd * np.tan(np.radians(exp_2theta))

    corrected_exp_2theta = q_to_2theta(q_corrected, exp_energy)
    
    # 보정된 SDD 계산 (실험 에너지 기준)
    corrected_sdd = original_r / np.tan(np.radians(corrected_exp_2theta))
    
    return corrected_sdd

def main():
    # 상수 정의
    ENERGY_CUKA = 8.042  # keV (Cu Kα)
    PIXEL_SIZE = 0.0886  # 픽셀 크기 (mm)
    BEAM_CENTER_X, BEAM_CENTER_Y = 955.1370, 633.0930  # 빔 센터 좌표
    IMAGE_SIZE = (1920, 1920)  # 이미지 크기
    EXP_ENERGY = 19.78  # keV (X선 에너지)
    MAX_CUKA_2THETA = 85.405622
    EXP_SDD = 227.7524  # mm
    
    # 테스트 1: q에서 2theta로 변환
    q_test = 3.0863  # Å^-1
    theta_2 = q_to_2theta(q_test, ENERGY_CUKA)
    print(f"\n테스트 1: q -> 2theta 변환")
    print(f"{q_test} Å^-1 -> {theta_2:.6f}°")
    
    # 테스트 2: 2theta에서 q로 변환
    theta_2_test = 44.50  # degree
    q_converted = theta_to_q(theta_2_test, ENERGY_CUKA)
    print(f"\n테스트 2: 2theta -> q 변환")
    print(f"{theta_2_test}° -> {q_converted:.6f} Å^-1")
    
    # 테스트 3: q_max 계산
    q_max = theta_to_q(MAX_CUKA_2THETA, ENERGY_CUKA)
    print(f"\n테스트 3: q_max 계산")
    print(f"q_max: {q_max:.6f} Å^-1")
    
    # 테스트 4: theta_2_max 계산
    theta_2_max = q_to_2theta(q_max, EXP_ENERGY)
    print(f"\n테스트 4: theta_2_max 계산")
    print(f"theta_2_max: {theta_2_max:.6f}°")
    
    # 테스트 5: 최대 SDD 계산
    SDD_max, R_mm_max = max_SDD_calculation(theta_2_max, PIXEL_SIZE, 
                                          BEAM_CENTER_X, BEAM_CENTER_Y, 
                                          IMAGE_SIZE)
    print(f"\n테스트 5: 최대 SDD 계산")
    print(f"Maximum SDD: {SDD_max:.4f} mm")
    print(f"Maximum R: {R_mm_max:.4f} mm")

    SDD_error = SDD_max - EXP_SDD

    print(f"Error: {SDD_error:.4f} mm")    

    # 테스트 6: SDD 보정 계산
    original_2theta_cuka = 42.7187489673154  # 예시 값, CuKα에서의 2θ 값
    corrected_2theta = 43.1060137546894     # 예시 값, 보정하고자 하는 2θ 값
    corrected_sdd = calculate_corrected_sdd(
        original_2theta_cuka,
        EXP_SDD,
        corrected_2theta,
        EXP_ENERGY
    )
    print(f"\n테스트 6: SDD 보정 계산")
    print(f"Original 2θ (Cu Kα): {original_2theta_cuka}°")
    print(f"Corrected 2θ: {corrected_2theta}°")
    print(f"Corrected SDD: {corrected_sdd:.4f} mm")



if __name__ == "__main__":
    main()
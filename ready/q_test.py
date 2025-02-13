import numpy as np

def q_to_2theta(q, energy_keV):
    """
    q (Å⁻¹)를 주어진 에너지(keV)에서의 2θ (°)로 변환하는 함수.
    
    Parameters
    ----------
    q : float or array
        q 값 (Å⁻¹)
    energy_keV : float
        X-ray 에너지 (keV)
    
    Returns
    -------
    float or array
        변환된 2θ 값 (°)
    """
    wavelength = 12.398 / energy_keV  # λ = 12.398 / E
    theta_rad = np.arcsin((q * wavelength) / (4 * np.pi))
    theta_deg = np.degrees(2 * theta_rad)
    return theta_deg

def theta_to_q(theta_2, energy_keV):
    """
    2θ (°)를 주어진 에너지(keV)에서의 q (Å⁻¹)로 변환하는 함수.
    
    Parameters
    ----------
    theta_2 : float or array
        2θ 값 (°)
    energy_keV : float
        X-ray 에너지 (keV)
    
    Returns
    -------
    float or array
        변환된 q 값 (Å⁻¹)
    """
    wavelength = 12.398 / energy_keV
    theta_rad = np.radians(theta_2 / 2)
    q_value = (4 * np.pi / wavelength) * np.sin(theta_rad)
    return q_value

def max_SDD_calculation(theta_2_max, pixel_size, beam_center_x, beam_center_y, image_size):
    """
    이미지의 네 모서리를 기준으로, 최대 방사 거리(R)를 구한 후 이를 이용하여
    최대 SDD (sample-to-detector distance)를 계산하는 함수.
    
    Parameters
    ----------
    theta_2_max : float
        최대 2θ 값 (°)
    pixel_size : float
        픽셀 크기 (mm)
    beam_center_x : float
        빔 센터 x 좌표 (MATLAB 좌표 기준)
    beam_center_y : float
        빔 센터 y 좌표 (MATLAB 좌표 기준)
    image_size : tuple
        이미지 크기 (width, height)
    
    Returns
    -------
    tuple
        (최대 SDD (mm), 최대 R (mm))
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
    """
    [단일 포인트용] 원래 데이터(원래 2θ 값 또는 q 값)를 기반으로,
    SDD 보정 후의 새로운 SDD 값을 계산한다.
    
    데이터 체계:
      - 만약 converted_energy가 None이면, 원본 데이터는 q 값으로 저장됨.
      - converted_energy가 주어지면, 원본 데이터는 변환 전 CuKα 2θ 값으로 저장됨.
    
    절차
    -------
    1) 입력 데이터(원래 2θ 또는 q)를, 만약 converted_energy가 주어졌다면 CuKα 기준의 2θ → q로 변환.
    2) 해당 q 값을 exp_energy 기준의 2θ로 변환.
    3) 기존 SDD를 이용하여, 각 포인트의 detector 상 방사거리 R = original_sdd * tan(2θ_exp)를 계산.
    4) 같은 R에서, 보정 SDD를 적용하면 새로운 2θ (exp_energy 기준)는 arctan(R/newSDD)로 구해짐.
    5) 이를 다시 q (exp_energy 기준)로 변환.
       - 만약 converted_energy가 주어지면 최종 결과를 다시 CuKα 2θ (즉, theta_to_q와 q_to_2theta를 거쳐)
         변환하여 출력한다.
    
    Parameters
    ----------
    original_2theta : float
        원래 데이터 값; converted_energy가 None이면 q 값, 아니면 CuKα 기준 2θ (°)
    original_sdd : float
        원래 SDD (mm)
    corrected_2theta : float
        보정하고자 하는 2θ 값; converted_energy가 None이면 q 값, 아니면 CuKα 기준 2θ (°)
    exp_energy : float
        실험 X-ray 에너지 (keV) – R 계산 시 사용
    converted_energy : float, optional
        입력 데이터가 변환된 CuKα 에너지 기준 값 (keV). 
        None이면 입력/출력은 q 값로 처리함.
    
    Returns
    -------
    corrected_sdd : float
        보정 후의 SDD (mm), 계산은 exp_energy 기준 2θ를 사용.
    """
    if converted_energy is None:
        # 입력 데이터가 q 값으로 저장됨
        q_original = original_2theta
        q_corrected = corrected_2theta
    else:
        # 입력 데이터가 CuKα 기준의 2θ 값으로 저장됨 → q로 변환
        q_original = theta_to_q(original_2theta, converted_energy)
        q_corrected = theta_to_q(corrected_2theta, converted_energy)
    
    # exp_energy 기준 2θ 계산
    exp_2theta = q_to_2theta(q_original, exp_energy)
    
    # 원래 SDD에서의 detector 반경 R 계산
    original_r = original_sdd * np.tan(np.radians(exp_2theta))
    
    # 보정된 2θ (exp_energy 기준) 계산: R = newSDD * tan(2θ_new)
    corrected_exp_2theta = q_to_2theta(q_corrected, exp_energy)
    
    # 보정 후 SDD = R / tan(2θ_new)
    corrected_sdd = original_r / np.tan(np.radians(corrected_exp_2theta))
    
    return corrected_sdd

def recalc_q_list(
    q_list,        
    original_sdd,  
    corrected_sdd, 
    energy_keV,     
    converted_energy=8.042  
):
    """
    [배열 단위] 원래 데이터(저장형태에 따라 q 값 또는 CuKα 기준 2θ 값)를,
    기존 SDD(original_sdd)에서 측정된 것으로부터 보정 SDD(corrected_sdd)를 적용할 경우
    최종적으로 exp_energy 기준으로 얻어야 하는 데이터(출력 형식은 입력과 동일)를 재계산한다.
    
    데이터 체계
    -----------
    - 만약 converted_energy가 None이면, 입력 q_list는 q 값이며 최종 결과도 q 값이다.
    - converted_energy가 주어지면, 입력 q_list는 CuKα 기준의 2θ 값(°)으로 저장되어 있으며,
      최종 결과도 동일한 방식(즉, 2θ 값)으로 출력한다.
    
    계산 절차
    -----------
    1) (converted_energy가 주어졌다면) 입력 2θ(CuKα) → q (CuKα)로 변환.
    2) 위 q 값을 exp_energy 기준의 2θ로 변환.
    3) 원래 SDD를 사용하여, 각 포인트의 반경 R = original_sdd * tan( exp_energy 기준 2θ ) 계산.
    4) 같은 R에 대해, 보정 SDD를 적용하면 새로운 2θ (exp_energy 기준)는 arctan(R / corrected_sdd)로 구해짐.
    5) 이 새로운 2θ를 exp_energy 기준의 q로 변환.
    6) 만약 converted_energy가 주어졌다면, 최종 결과를 다시 CuKα 기준의 2θ 값으로 변환.
    
    Parameters
    ----------
    q_list : array-like
        원래 데이터. 
        - converted_energy가 None이면 q 값 (Å⁻¹)
        - converted_energy가 주어지면 CuKα 기준의 2θ 값 (°)
    original_sdd : float
        원래 SDD (mm)
    corrected_sdd : float
        보정된 SDD (mm)
    energy_keV : float
        exp_energy (q 또는 2θ 변환 시 사용되는 에너지, keV)
    converted_energy : float, optional
        입력 데이터가 CuKα 기준일 경우의 에너지 (keV). 
        None이면 변환 없이 q 값으로 처리.
    
    Returns
    -------
    corrected_q_list : ndarray
        - converted_energy가 None이면: 보정 후 q 값 (Å⁻¹)
        - converted_energy가 주어지면: 보정 후 CuKα 기준의 2θ 값 (°)
    """
    # 만약 converted_energy가 주어졌다면, 입력 데이터는 2θ 값(°)이므로 q로 변환
    if converted_energy is None:
        # 입력이 q 값으로 저장됨 → 별도 변환 없이 사용
        q_conv = q_list
    else:
        q_conv = theta_to_q(q_list, converted_energy)
    
    # 1) exp_energy 기준의 2θ 값으로 변환
    exp_2theta = q_to_2theta(q_conv, energy_keV)
    
    # 2) 원래 SDD에서의 반경 R 계산
    R = original_sdd * np.tan(np.radians(exp_2theta))
    
    # 3) 보정 SDD 적용 시의 새로운 2θ (exp_energy 기준)
    corrected_2theta = np.degrees(np.arctan(R / corrected_sdd))
    
    # 4) 새로운 2θ를 exp_energy 기준의 q로 변환
    q_temp = theta_to_q(corrected_2theta, energy_keV)
    
    # 5) 만약 converted_energy가 주어졌다면, 최종 결과를 CuKα 기준의 2θ 값으로 변환
    if converted_energy is None:
        corrected_q_list = q_temp
    else:
        corrected_q_list = q_to_2theta(q_temp, converted_energy)
    
    return corrected_q_list

def main():
    # 상수 정의
    ENERGY_CUKA = 8.042      # CuKα 에너지 (keV)
    PIXEL_SIZE = 0.0886      # 픽셀 크기 (mm)
    BEAM_CENTER_X, BEAM_CENTER_Y = 955.1370, 633.0930  # 빔 센터 좌표 (MATLAB 기준)
    IMAGE_SIZE = (1920, 1920)    # 이미지 크기 (width, height)
    EXP_ENERGY = 19.78       # 실험 X선 에너지 (keV)
    MAX_CUKA_2THETA = 85.405622  # 최대 2θ (°) [CuKα 기준]
    EXP_SDD = 227.7524       # 원래 SDD (mm)
    
    # 테스트 1: q -> 2θ 변환 (CuKα 에너지 기준)
    q_test = 3.0863  # Å⁻¹
    theta_2 = q_to_2theta(q_test, ENERGY_CUKA)
    print(f"\n테스트 1: q -> 2θ 변환")
    print(f"{q_test} Å⁻¹ -> {theta_2:.6f}°")
    
    # 테스트 2: 2θ -> q 변환 (CuKα 에너지 기준)
    theta_2_test = 44.50  # °
    q_converted = theta_to_q(theta_2_test, ENERGY_CUKA)
    print(f"\n테스트 2: 2θ -> q 변환")
    print(f"{theta_2_test}° -> {q_converted:.6f} Å⁻¹")
    
    # 테스트 3: q_max 계산 (CuKα 에너지 기준)
    q_max = theta_to_q(MAX_CUKA_2THETA, ENERGY_CUKA)
    print(f"\n테스트 3: q_max 계산")
    print(f"q_max: {q_max:.6f} Å⁻¹")
    
    # 테스트 4: 2θ_max 계산 (exp_energy 기준)
    theta_2_max = q_to_2theta(q_max, EXP_ENERGY)
    print(f"\n테스트 4: 2θ_max 계산")
    print(f"2θ_max: {theta_2_max:.6f}°")
    
    # 테스트 5: 최대 SDD 계산
    SDD_max, R_mm_max = max_SDD_calculation(theta_2_max, PIXEL_SIZE, BEAM_CENTER_X, BEAM_CENTER_Y, IMAGE_SIZE)
    print(f"\n테스트 5: 최대 SDD 계산")
    print(f"Maximum SDD: {SDD_max:.4f} mm")
    print(f"Maximum R: {R_mm_max:.4f} mm")
    SDD_error = SDD_max - EXP_SDD
    print(f"Error: {SDD_error:.4f} mm")
    
    # 테스트 6: SDD 보정 계산
    # 여기서 입력은 CuKα 기준의 2θ 값 (°)
    original_2theta_cuka = 42.7187489673154  # 예시값 (°)
    corrected_2theta = 43.1060137546894       # 예시값 (°)
    corrected_sdd = calculate_corrected_sdd(
        original_2theta_cuka,
        EXP_SDD,
        corrected_2theta,
        EXP_ENERGY
    )
    print(f"\n테스트 6: SDD 보정 계산")
    print(f"Original 2θ (CuKα): {original_2theta_cuka}°")
    print(f"Corrected 2θ (CuKα): {corrected_2theta}°")
    print(f"Calculated Corrected SDD: {corrected_sdd:.4f} mm")
    
    # 테스트 7: recalc_q_list 사용 예시
    # 입력: CuKα 기준의 2θ 값 배열 (0° ~ 85.405622°, 100포인트)
    cuka_2theta_array = np.linspace(0, 85.405622, 100)
    # 보정 후의 결과는, 데이터 체계에 따라:
    # - converted_energy가 지정되었으므로 입력과 동일하게 2θ (CuKα 기준)로 반환됨.
    corrected_q_list = recalc_q_list(
        q_list=cuka_2theta_array,
        original_sdd=EXP_SDD,
        corrected_sdd=corrected_sdd,
        energy_keV=EXP_ENERGY,
        converted_energy=ENERGY_CUKA
    )
    print("\n[테스트] recalc_q_list 사용 예시")
    print(f"입력 (CuKα 2θ) 범위: {cuka_2theta_array[0]:.6f}° ~ {cuka_2theta_array[-1]:.6f}°")
    print(f"보정 후 (CuKα 2θ) 범위: {corrected_q_list[0]:.6f}° ~ {corrected_q_list[-1]:.6f}°")

if __name__ == "__main__":
    main()
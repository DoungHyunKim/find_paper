import cv2
import numpy as np
import os

def adjust_image_resolution(image: np.ndarray, target_dpi: int = 300) -> np.ndarray:
    """
    이미지 해상도를 조정하고 최적화합니다.
    """
    # 이미지 크기 계산
    height, width = image.shape[:2]

    # A4 용지 기준 최적 크기 계산 (300DPI 기준)
    target_width = int(8.27 * target_dpi)  # A4 width in inches * DPI

    # 비율 유지하면서 리사이즈
    ratio = target_width / width
    new_width = int(width * ratio)
    new_height = int(height * ratio)

    # 이미지 리사이즈
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    return resized

def process_image_files(image_paths):
    # 이미지가 1개일 경우 처리
    if isinstance(image_paths, str):
        image_paths = [image_paths]  # 문자열을 리스트로 변환

    # 이미지 목록 처리
    for image_path in image_paths:
        # 이미지 경로 확인
        if not os.path.exists(image_path):
            print(f"파일을 찾을 수 없습니다: {image_path}")
            continue

        image = cv2.imread(image_path)

        if image is None:
            print(f"이미지를 로드할 수 없습니다: {image_path}")
            continue

        # 해상도 조정
        resized_image = adjust_image_resolution(image)

        # 결과를 파일로 저장
        resized_image_path = f"resized_{os.path.basename(image_path)}"
        cv2.imwrite(resized_image_path, resized_image)
        print(f"저장 완료: {resized_image_path}")

        # 결과 화면에 출력 (원하는 경우)
        cv2.imshow("Resized Image", resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# 예시: 1개의 이미지 경로
image_path = "test1.JPG"
process_image_files(image_path)

# 예시: 여러 개의 이미지 경로
image_paths = [
    "test1.JPG", "test2.JPG", "test3.JPG",
    "test4.JPG", "test5.JPG", "test6.JPG", "test7.JPG",]
process_image_files(image_paths)
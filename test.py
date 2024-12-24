# import cv2
# import matplotlib.pyplot as plt # 결과 확인을 위한 matplotlib 추가
#
# def load_image(image_path):
#     """이미지 파일 경로를 받아 이미지를 로드합니다."""
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"Error: Could not open or read image file: {image_path}")
#         return None
#     return image
#
# def convert_to_hsv(image):
#     """RGB 이미지를 HSV 색상 공간으로 변환합니다."""
#     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     return hsv_image
#
# def convert_to_lab(image):
#     """RGB 이미지를 Lab 색상 공간으로 변환합니다."""
#     lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#     return lab_image
#
# def enhance_contrast_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
#     """CLAHE를 사용하여 명암 대비를 향상시킵니다."""
#     lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) # Lab 색상 공간에서 L 채널만 처리
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
#     cl = clahe.apply(l)
#     enhanced_lab = cv2.merge((cl, a, b))
#     enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
#     return enhanced_image
#
# def remove_noise_gaussian(image, kernel_size=(5, 5)):
#     """가우시안 블러링으로 노이즈를 제거합니다."""
#     blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
#     return blurred_image
#
# def remove_noise_median(image, kernel_size=5):
#     """미디안 블러링으로 노이즈를 제거합니다."""
#     blurred_image = cv2.medianBlur(image, kernel_size)
#     return blurred_image
#
# def binarize_adaptive(image, block_size=11, c=2):
#     """적응적 임계값 처리를 사용하여 이진화합니다."""
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     binary_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c)
#     return binary_image
#
# def binarize_otsu(image):
#     """Otsu's method를 사용하여 이진화합니다."""
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     return binary_image
#
# # 이미지 불러오기 (경로 수정 필요)
# image_path = "test1.JPG"  # 실제 이미지 파일 경로로 변경
# original_image = load_image(image_path)
#
# if original_image is not None:
#     # matplotlib을 사용하여 이미지 표시 (색상 반전 주의)
#     plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
#     plt.title("Original Image")
#     plt.show()
#
#     hsv_image = convert_to_hsv(original_image)
#     lab_image = convert_to_lab(original_image)
#
#     # HSV 이미지의 H, S, V 채널 분리 및 표시
#     h, s, v = cv2.split(hsv_image)
#     plt.figure(figsize=(12, 4))
#     plt.subplot(131), plt.imshow(h, cmap='gray'), plt.title("H Channel") # 색상: 색상환에서의 위치를 나타냄
#     plt.subplot(132), plt.imshow(s, cmap='gray'), plt.title("S Channel") # 채도: 색의 순도를 나타냄. 채도가 높을 수록 선명
#     plt.subplot(133), plt.imshow(v, cmap='gray'), plt.title("V Channel") # 명도: 색의 밝기를 나타냄
#     plt.show()
#
#     # Lab 이미지의 L, a, b 채널 분리 및 표시
#     l, a, b = cv2.split(lab_image)
#     plt.figure(figsize=(12, 4))
#     plt.subplot(131), plt.imshow(l, cmap='gray'), plt.title("L Channel")  # 밝기를 나타낸다.
#     plt.subplot(132), plt.imshow(a, cmap='gray'), plt.title("a Channel")  # 빨강-녹색 축을 나타냄
#     plt.subplot(133), plt.imshow(b, cmap='gray'), plt.title("b Channel")  # 노랑-파랑 축을 나타냄
#     plt.show()
#
#     # 명암 대비 조절
#     clahe_image = enhance_contrast_clahe(original_image)
#     plt.imshow(cv2.cvtColor(clahe_image, cv2.COLOR_BGR2RGB))
#     plt.title("CLAHE Enhanced Image")
#     plt.show()
#
#     # 노이즈 제거
#     gaussian_blurred = remove_noise_gaussian(original_image)
#     median_blurred = remove_noise_median(original_image)
#
#     plt.figure(figsize=(12, 4))
#     plt.subplot(121), plt.imshow(cv2.cvtColor(gaussian_blurred, cv2.COLOR_BGR2RGB)), plt.title("Gaussian Blurred")
#     plt.subplot(122), plt.imshow(cv2.cvtColor(median_blurred, cv2.COLOR_BGR2RGB)), plt.title("Median Blurred") # 이게 더 좋아보임.
#     plt.show()
#
#     # 이진화
#     adaptive_binary = binarize_adaptive(original_image)
#     otsu_binary = binarize_otsu(original_image)
#
#     plt.figure(figsize=(12, 4))
#     plt.subplot(121), plt.imshow(adaptive_binary, cmap='gray'), plt.title("Adaptive Thresholding")
#     plt.subplot(122), plt.imshow(otsu_binary, cmap='gray'), plt.title("Otsu's Method") # 밝기 조절해주면 괜찮을 수도, 부분적으로 나누어서 필터를 적용할 수 는 없나?
#     plt.show()



## 초 대박 잘 나옴
# import cv2
# import matplotlib.pyplot as plt
#
#
# def preprocess_image(image_path, clahe_clip_limit=2.0, clahe_tile_grid_size=(8, 8), gaussian_kernel_size=(5, 5),
#                      adaptive_block_size=11, adaptive_c=2):
#     """이미지를 전처리합니다."""
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"Error: Could not open or read image file: {image_path}")
#         return None
#
#     # CLAHE를 이용한 명암 대비 향상
#     lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
#     cl = clahe.apply(l)
#     enhanced_lab = cv2.merge((cl, a, b))
#     image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
#
#     # 가우시안 블러링을 이용한 노이즈 제거
#     image = cv2.GaussianBlur(image, gaussian_kernel_size, 0)
#
#     # 이진화 (Otsu's method)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#
#     return binary_image
#
#
# # 이미지 경로 (실제 경로로 변경 필요)
# image_path = "test2.JPG"
# preprocessed_image = preprocess_image(image_path)
#
# if preprocessed_image is not None:
#     plt.figure(figsize=(10, 5))
#     plt.subplot(121), plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)), plt.title("Original Image")
#     plt.subplot(122), plt.imshow(preprocessed_image, cmap='gray'), plt.title("Preprocessed Image")
#     plt.show()



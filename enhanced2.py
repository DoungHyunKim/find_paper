import cv2
import numpy as np


def adjust_brightness(img, value):
    """
    이미지 밝기를 조절하는 함수

    Args:
      img: 입력 이미지
      value: 밝기 조절 값 (-255 ~ 255)

    Returns:
      밝기가 조절된 이미지
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    v = cv2.add(v, value)
    v = np.clip(v, 0, 255).astype(np.uint8)

    hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def find_corners(lines, img_shape):
    """
    직선들의 교점을 계산하여 이미지의 네 모서리를 찾는 함수

    Args:
      lines: HoughLinesP 함수로 찾은 직선 리스트
      img_shape: 이미지 shape (height, width, channels)

    Returns:
      이미지 네 모서리 좌표 리스트
    """

    if lines is None:
        raise ValueError("No lines detected.")

    # 교점 찾기
    intersections = []
    for i, line1 in enumerate(lines):
        for j, line2 in enumerate(lines):
            if i >= j:
                continue
            x1, y1, x2, y2 = line1[0]
            x3, y3, x4, y4 = line2[0]

            # 두 직선의 기울기 계산
            m1 = (y2 - y1) / (x2 - x1) if x2 != x1 else np.inf
            m2 = (y4 - y3) / (x4 - x3) if x4 != x3 else np.inf

            # 기울기가 같은 직선(평행) 제외
            if m1 == m2:
                continue

            # 교점 계산
            if m1 == np.inf:
                x = x1
                y = m2 * x + (y3 - m2 * x3)
            elif m2 == np.inf:
                x = x3
                y = m1 * x + (y1 - m1 * x1)
            else:
                x = (y3 - y1 + m1 * x1 - m2 * x3) / (m1 - m2)
                y = m1 * x + (y1 - m1 * x1)

            # 이미지 경계 내에 있는 교점만 저장
            if 0 <= x < img_shape[1] and 0 <= y < img_shape[0]:
                intersections.append((int(x), int(y)))

    if len(intersections) < 4:
        raise ValueError("Not enough valid corners found.")

    # y좌표, x좌표 순으로 정렬하여 좌상단, 좌하단, 우상단, 우하단 순서로 정렬
    intersections = sorted(intersections, key=lambda p: (p[1], p[0]))

    # 네 모서리 반환
    return intersections[:4]


def detect_paper(image):
    """
    색상 정보와 Hough 변환을 사용하여 이미지에서 용지를 인식하는 함수

    Args:
      image: 입력 이미지

    Returns:
      용지 영역의 좌표 (x, y, w, h) 또는 None
    """
    # 밝기 조절
    adjusted_image = adjust_brightness(image, 50)  # 밝기 조절 값은 적절히 조정

    # 대비 조절 (필요에 따라 주석 처리)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # adjusted_image = clahe.apply(cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY))

    # HSV 색 공간으로 변환
    hsv = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2HSV)

    # 흰색 범위 지정 (필요에 따라 범위 조정)
    lower_white = np.array([0, 0, 150])
    upper_white = np.array([180, 50, 255])

    # 흰색 영역 마스크 생성
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # 엣지 검출
    edges = cv2.Canny(mask, 50, 150)

    # Hough 변환
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

    # 직선 분석 및 용지 꼭짓점 찾기
    try:
        corners = find_corners(lines, image.shape)
    except ValueError as e:
        print(f"Error: {e}")
        return None

    # 꼭짓점을 순서대로 정렬 (좌상, 좌하, 우하, 우상)
    if corners is not None:
        # Perspective Transform 적용
        pts1 = np.float32(corners)
        width = max(np.linalg.norm(pts1[0] - pts1[1]), np.linalg.norm(pts1[2] - pts1[3]))
        height = max(np.linalg.norm(pts1[1] - pts1[2]), np.linalg.norm(pts1[3] - pts1[0]))
        pts2 = np.float32([[0, 0], [0, height], [width, height], [width, 0]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        transformed_image = cv2.warpPerspective(adjusted_image, matrix, (int(width), int(height)))

        # 변환된 이미지에서 용지 영역 다시 인식 (필요시 추가적인 처리)
        # ...

        return cv2.boundingRect(np.array(corners))

    return None  # 용지 인식 실패 시 None 반환


def detect_lines(image_path, brightness_value=50):  # 밝기 조절 값을 인자로 받음
    """
    이미지에서 직선과 용지를 검출하는 함수

    Args:
      image_path: 이미지 파일 경로
      brightness_value: 밝기 조절 값 (-255 ~ 255)
    """
    # 이미지 로드
    image = cv2.imread(image_path)

    # 용지 인식
    paper_rect = detect_paper(image)

    if paper_rect is not None:
        x, y, w, h = paper_rect
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 파란색 사각형으로 용지 영역 표시

    # 결과 이미지 출력
    cv2.imshow("Detected Lines and Corners", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 이미지 경로 지정
image_path = "test7.jpg"
brightness_value = 50
detect_lines(image_path, brightness_value)
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
    # 직선 필터링 (길이 기준)
    lines = [line for line in lines if
             np.linalg.norm(np.array(line[0][:2]) - np.array(line[0][2:])) > img_shape[1] * 0.1]  # 가로 길이의 10% 이상인 직선만 선택

    # 직선 필터링 (기울기 기준)
    def get_angle(line):
        x1, y1, x2, y2 = line[0]
        return np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

    lines = [line for line in lines if abs(get_angle(line)) < 10 or abs(get_angle(line)) > 80]  # 기울기가 10도 이하 또는 80도 이상인 직선만 선택

    intersections = []
    for i, line1 in enumerate(lines):
        for j, line2 in enumerate(lines):
            if i >= j:
                continue

            # Extract coordinates
            x1, y1, x2, y2 = line1[0]
            x3, y3, x4, y4 = line2[0]

            # Compute slopes and intercepts
            m1 = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else np.inf
            c1 = y1 - m1 * x1 if m1 != np.inf else np.inf

            m2 = (y4 - y3) / (x4 - x3) if (x4 - x3) != 0 else np.inf
            c2 = y3 - m2 * x3 if m2 != np.inf else np.inf

            # Skip parallel lines
            if m1 == m2:
                continue

            # Compute intersection
            if m1 == np.inf:  # Line1 vertical
                x = x1
                y = m2 * x + c2
            elif m2 == np.inf:  # Line2 vertical
                x = x3
                y = m1 * x + c1
            else:
                x = (c2 - c1) / (m1 - m2)
                y = m1 * x + c1

            # Check if the intersection is within image bounds
            if 0 <= x < img_shape[1] and 0 <= y < img_shape[0]:
                intersections.append((int(x), int(y)))

    if len(intersections) < 4:
        raise ValueError("Not enough valid corners found.")

    # 디버깅
    print(f"Found {len(intersections)} intersections.")

    intersections = sorted(intersections, key=lambda p: (p[1], p[0]))
    return intersections[:4]  # 네 모서리


def detect_paper(image):
    """
    이미지에서 용지를 인식하는 함수

    Args:
      image: 입력 이미지

    Returns:
      용지 영역의 좌표 (x, y, w, h)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200) # 1. 75/200 2. 50/200

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return cv2.boundingRect(approx)

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

    # 밝기 조절 적용
    image = adjust_brightness(image, brightness_value)

    # 용지 인식
    paper_rect = detect_paper(image)

    if paper_rect is not None:
        x, y, w, h = paper_rect
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 파란색 사각형으로 용지 영역 표시

    # 그레이스케일로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 노이즈 제거( 미디언 필터 적용)
    gray = cv2.medianBlur(gray, 5)  # 5x5 커널 크기의 미디언 필터 적용

    # Canny 엣지 검출
    edges = cv2.Canny(gray, 50, 150)

    # 이미지 크기 분석
    height, width = edges.shape
    diagonal_length = int(np.sqrt(height ** 2 + width ** 2))  # 이미지의 대각선 길이

    # minLineLength: 전체 길이에 비례하도록 설정
    min_line_length = int(0.05 * diagonal_length)  # 대각선의 5% 정도로 설정 (기존 2%에서 증가)

    # maxLineGap: 전체 폭과 비례
    max_line_gap = int(0.01 * width)  # 가로 길이의 1% 정도로 설정

    # threshold: 엣지 밀도에 따라 조정 (경험적인 값 사용)
    non_zero_edges = cv2.countNonZero(edges)  # 엣지가 있는 픽셀 수
    edge_density = non_zero_edges / (height * width)
    if edge_density > 0.05:  # 엣지가 많을 경우 높은 threshold
        threshold = 150  # 기존 100에서 증가
    else:  # 엣지가 적을 경우 낮은 threshold
        threshold = 80  # 기존 50에서 증가

    # 확장 허프 변환(직선 검출)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold,
                            minLineLength=min_line_length,
                            maxLineGap=max_line_gap)

    # 코너 찾기
    try:
        corners = find_corners(lines, image.shape)

        # 코너 표시
        for corner in corners:
            cv2.circle(image, corner, 10, (0, 0, 255), -1)  # 빨간색 점으로 코너 표시
    except ValueError as e:
        print(f"Error: {e}")

    # 직선이 발견되었으면
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 이미지에 직선 그리기
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 초록색 선

    # 결과 이미지 출력
    cv2.imshow("Detected Lines and Corners", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 이미지 경로 지정
image_path = "test1.jpg"
brightness_value = 50
detect_lines(image_path, brightness_value)
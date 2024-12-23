from itertools import count

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
    색상 정보와 윤곽선 검출을 사용하여 이미지에서 흰색 포스트잇을 인식하는 함수

    Args:
      image: 입력 이미지

    Returns:
      변환된 포스트잇 이미지 또는 None
    """

    # HSV 색 공간으로 변환
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 흰색 범위 지정 (필요에 따라 범위 조정)
    lower_white = np.array([0, 0, 180])  # 흰색 범위 하한
    upper_white = np.array([180, 50, 255])  # 흰색 범위 상한

    # 흰색 영역 마스크 생성
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # 모폴로지 연산을 사용하여 노이즈 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 윤곽선 검출
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # 디버깅
    # print("mask shape:", mask.shape)  # 마스크 크기 출력
    # print("mask unique values:", np.unique(mask))  # 마스크에 포함된 유일한 값 출력 (0 또는 255)
    # cv2.imshow("mask", mask)  # 마스크 이미지 출력
    # cv2.waitKey(0)
    # print("number of contours:", len(cnts))  # 윤곽선 개수 출력
    # for i, cnt in enumerate(cnts):
    #     print(f"contour {i} area:", cv2.contourArea(cnt))  # 각 윤곽선의 면적 출력


    if cnts:
        # 가장 큰 윤곽선 찾기
        cnt = max(cnts, key=cv2.contourArea)

        # 윤곽선 근사 및 꼭짓점 찾기
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

        # 디버깅용
        # print(count(approx))

        if len(approx) == 4:
            # 이미지 변환: 특정 시점에서 바라본 것처럼 변환한다. Perspective Transform 행렬을 사용
            pts1 = np.float32(approx)
            pts2 = np.float32([[0, 0], [0, 300], [400, 300], [400, 0]])  # 변환될 좌표 (크기는 임의로 설정)
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            transformed_image = cv2.warpPerspective(image, matrix, (400, 300))

            # 노이즈 제거
            # transformed_image = cv2.fastNlMeansDenoisingColored(transformed_image, None, 5, 5, 7, 21)

            # 샤프닝 필터 적용
            # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            # transformed_image = cv2.filter2D(transformed_image, -1, kernel)

            # 변환된 이미지에서 용지 영역 파란색으로 표시(디버깅용)
            cv2.drawContours(image, [approx], -1, (255, 0, 0), 3)  # 파란색으로 윤곽선 그림

            # 이미지 크기 조정 (동적 크기 조정)
            height, width = transformed_image.shape[:2]
            target_width = 1000  # 예시: 가로 길이 1000 픽셀을 기준으로 조정
            if width > target_width:
                new_width = target_width
                new_height = int(height * (target_width / width))
                transformed_image = cv2.resize(transformed_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

            return transformed_image
            # 디버깅용
            # print("corners:", approx)  # 네 꼭짓점 좌표 출력
            # print("transformed image shape:", transformed_image.shape)  # 변환된 이미지 크기 출력

            return transformed_image

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

    # 디비거깅용
    # print(image.shape)

    # 밝기 조절 적용
    image = adjust_brightness(image, brightness_value)

    # 용지 인식
    transformed_image = detect_paper(image)  # 변환된 이미지를 받아옴

    # 디비거깅용
    # print("transformed_image:", transformed_image)

    if transformed_image is not None:
        # 변환된 이미지 출력
        cv2.imshow("Transformed Image", transformed_image)  # 변환된 이미지 출력

        # 원본 이미지에 용지 영역 표시 (디버깅용)
        cv2.imshow("Original Image with Paper", image)  # 원본 이미지 출력
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# 이미지 경로 지정
image_path = "test7.JPG"
brightness_value = 50
detect_lines(image_path, brightness_value)
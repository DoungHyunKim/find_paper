import cv2
import numpy as np

def enhance_resolution(image_path):
    """
    이미지에서 용지 영역을 찾아 해상도를 개선하고 저장합니다.
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 75, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=10, maxLineGap=10)

    if lines is None or len(lines) < 4:
        raise ValueError("4개 이상의 직선을 찾을 수 없습니다.")

    # 4개의 직선을 찾기 위한 처리
    lines = lines.squeeze()
    slopes = [(y2 - y1) / (x2 - x1) for x1, y1, x2, y2 in lines]
    vertical_lines = lines[np.isinf(slopes)]
    horizontal_lines = lines[~np.isinf(slopes)]

    if len(vertical_lines) < 2 or len(horizontal_lines) < 2:
        raise ValueError("수직선과 수평선을 각각 2개 이상 찾을 수 없습니다.")

    # 교차점 찾기
    intersections = []
    for v_line in vertical_lines:
        for h_line in horizontal_lines:
            x1, y1, x2, y2 = v_line
            x3, y3, x4, y4 = h_line
            v_slope = (y2 - y1) / (x2 - x1)
            h_slope = (y4 - y3) / (x4 - x3)
            if not np.isclose(v_slope, h_slope):
                intersection_x = int((y3 - y1 + v_slope * x1 - h_slope * x3) / (v_slope - h_slope))
                intersection_y = int(v_slope * (intersection_x - x1) + y1)
                intersections.append((intersection_x, intersection_y))

    if len(intersections) != 4:
        raise ValueError("4개의 교차점을 찾을 수 없습니다.")

    # 꼭짓점 정렬
    intersections.sort(key=lambda p: (p[1], p[0]))  # y좌표 오름차순, x좌표 오름차순으로 정렬
    top_left, top_right, bottom_left, bottom_right = intersections

    # 변환 행렬 계산
    width = max(np.linalg.norm(np.array(top_right) - np.array(top_left)),
                 np.linalg.norm(np.array(bottom_right) - np.array(bottom_left)))
    height = max(np.linalg.norm(np.array(bottom_left) - np.array(top_left)),
                  np.linalg.norm(np.array(bottom_right) - np.array(top_right)))
    dst_pts = np.array([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]], dtype="float32")
    src_pts = np.array([top_left, top_right, bottom_left, bottom_right], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # 이미지 변환
    warped_img = cv2.warpPerspective(image, M, (int(width), int(height)))

    # 용지 영역 자르기
    cropped_image = warped_img[int(height * 0.05):int(height * 0.95), int(width * 0.05):int(width * 0.95)]

    # Super-resolution 적용 (OpenCV Dnn Super Res 예시)
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel("EDSR_x4.pb")
    sr.setModel("edsr", 4)
    result = sr.upsample(cropped_image)

    # 결과 저장
    cv2.imwrite("enhanced_image.jpg", result)

# 이미지 경로
image_path = "test1.JPG"

# 해상도 개선 함수 호출
enhance_resolution(image_path)
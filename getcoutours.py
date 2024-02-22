import cv2 as cv
import numpy as np
# from stackImages import stackImages

def get_area(area,S_pixel_1cm2):
    S = area / S_pixel_1cm2
    return S

# ảnh img là ảnh đã gray....-> dilated, imgContours nguyên trạng sẽ vẽ lên đó
def getContours(img,imgContours,areamin,areamax,S_none=0):
    'img là ảnh gốc, imgContours là ảnh coppy để không bị lỗi đè\n.   areamin là kích thước tối thiểu chập nhận được, areamax tối đa chập nhận được. \n trả về img,gray,imgCanny,dilated,img_cout'
    # tìm đường viền, contours danh sách các đường viền,hierarchy,mảng có cấu trúc, mô tả mối quan hệ giữa các đường viền.
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        #  tính diện tích của một đường viền (contour) 
        area = cv.contourArea(cnt)
        S = area
        # S = get_area(area, 2300)
        if areamin < S <= areamax:
            # vẽ các đường viền (contours) đã được tìm thấy
            cv.drawContours(imgContours, cnt, -1, (255, 0, 255),2)
            # chiều dài của một đường viền (contour) trong ảnh. T đóng, F hở
            peri = cv.arcLength(cnt, True)
            # xấp xỉ một đường viền (contour) bằng một đa giác (polygon) có số cạnh xác định
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
            # print(approx)
            # tính hình chữ nhật bao quanh một đối tượng trong ảnh.
            x, y, w, h = cv.boundingRect(approx)

            cv.rectangle(imgContours, (x, y), (x + w, y + h), (0, 255, 255), 2)

            # cv.putText(imgContours, "points " + str(len(approx)), (x + w + 20, y + 20), cv.FONT_HERSHEY_COMPLEX, 0.7,
            #            (255, 255, 255), 2)
            # cv.putText(imgContours, "area " + str(int(S)), (x + w + 20, y + 45), cv.FONT_HERSHEY_COMPLEX, 0.7,
            #            (255, 255, 255), 2)
            S_none = S
    return imgContours,S_none

def proceed(img,imgContours,threshold1,threshold2,areamin,areamax):
    'img là ảnh gốc, threshold1 ngưỡn phân ảnh, areamin < area < areamax'
    blur = cv.GaussianBlur(img, (5, 5), cv.BORDER_DEFAULT)
    
    gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    # closed = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel)
    # opened = cv.morphologyEx(closed, cv.MORPH_OPEN, kernel)
    # equalized_img = cv.equalizeHist(opened)

    imgCanny = cv.Canny(gray, threshold1, threshold2)
    kernel = np.ones((3, 3))
    dilated = cv.dilate(imgCanny, kernel, iterations=1)
    img_cout,S_nice = getContours(dilated,imgContours,areamin,areamax)
    return img,gray,imgCanny,dilated,img_cout,S_nice

# while True:

#     # success, img = cap.read()
#     img = cv.imread('D:/study/NCKH/opencv/NCKH/imgs/benh_den.jpg')
#     imgContours = img.copy()

#     blur = cv.GaussianBlur(img, (7, 7), cv.BORDER_DEFAULT)

#     gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)

#     kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
#     closed = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel)
#     opened = cv.morphologyEx(closed, cv.MORPH_OPEN, kernel)
#     # equalized_img = cv.equalizeHist(gray)

#     # Hiển thị ảnh gốc và ảnh đã cân bằng histogram
#     # cv.imshow('Original Image', img_ee)
#     # cv.imshow('Equalized Image', equalized_img)
#     threshold1 = cv.getTrackbarPos("Threshold1", "Parameters")
#     threshold2 = cv.getTrackbarPos("Threshold2", "Parameters")

#     imgCanny = cv.Canny(opened, threshold1, threshold2)

#     kernel = np.ones((5, 5))
#     dilated = cv.dilate(imgCanny, kernel, iterations=1)

#     areamin = cv.getTrackbarPos("Areamin", "Parameters")
#     areamax = cv.getTrackbarPos("Areamax", "Parameters")
#     img_cout = getContours(dilated, imgContours,areamin,areamax)

#     scale = cv.getTrackbarPos("scale", "Parameters")
#     imgStack = stackImages(scale/10, ([img, gray, imgCanny],
#                                 [dilated, imgContours, dilated]))
#     cv.imshow('Result', imgStack)
#     key = cv.waitKey(1)
#     if key == ord('q'):
#         break

# # Giải phóng bộ nhớ và đóng cửa sổ
# cap.release()
# cv.destroyAllWindows()


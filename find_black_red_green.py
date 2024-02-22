import cv2 as cv
import numpy as np
# from stackImages import stackImages
# from getcoutours import proceed
# import trackbar
# 4:3, 16:9, 3:2
# frameWith = 640
# frameHeight = 480
# cap = cv.VideoCapture(0)
# cap.set(3, frameWith)
# cap.set(4, frameHeight)

def dilate_mask(mask_red, iterations):
    # Tạo kernel hình chữ nhật có kích thước 3x3
    kernel = np.ones((5, 5), np.uint8)
    # Giãn vùng màu đỏ bằng phép toán dilation
    dilated_mask_red = cv.dilate(mask_red, kernel, iterations=iterations)
    return dilated_mask_red

def detect_one_color(img,array_lowr=[0,0,0],array_upper=[0,0,0],iterations=2):
    image_source = img.copy()
    hsv = cv.cvtColor(image_source, cv.COLOR_BGR2HSV)

    lower_green = np.array(array_lowr)
    upper_green = np.array(array_upper)

    mask = cv.inRange(hsv, lower_green, upper_green)

    img_dilated = dilate_mask(mask, iterations=iterations)

    img_detect_color = cv.bitwise_and(image_source, image_source, mask=img_dilated)
    return img_detect_color,img_dilated

def detect_two_color(img,array_lowr1=[0,0,0],array_upper1=[0,0,0],array_lowr2=[0,0,0],array_upper2=[0,0,0],iterations=2):
    image_source = img.copy()
    hsv = cv.cvtColor(image_source, cv.COLOR_BGR2HSV)

    lower1 = np.array(array_lowr1)
    upper1 = np.array(array_upper1)

    lower2 = np.array(array_lowr2)
    upper2 = np.array(array_upper2)

    mask1 = cv.inRange(hsv, lower1, upper1)
    mask2 = cv.inRange(hsv, lower2, upper2)

    mask = mask1 + mask2

    img_dilated = dilate_mask(mask, iterations=iterations)

    img_detect_color = cv.bitwise_and(image_source, image_source, mask=img_dilated)
    return img_detect_color,img_dilated

def detect_color(img, color_ranges, iterations=2,code: int = cv.COLOR_BGR2LAB):
    image_source = img.copy()
    image_blur = cv.GaussianBlur(image_source, (5, 5), 1)
    lab = cv.cvtColor(image_blur, code)

    # Khởi tạo mask toàn bộ là 0
    mask = np.zeros(image_source.shape[:2], dtype=np.uint8)

    # Áp dụng các khoảng màu vào mask
    for lower, upper in color_ranges:
        lower_bound = np.array(lower)
        upper_bound = np.array(upper)
        mask_part = cv.inRange(lab, lower_bound, upper_bound)
        mask = cv.bitwise_or(mask, mask_part)

    # Giãn vùng mask
    img_dilated = dilate_mask(mask, iterations=iterations)

    # Áp dụng mask lên ảnh gốc
    img_detect_color = cv.bitwise_and(image_source, image_source, mask=img_dilated)

    return img_detect_color, img_dilated

# # ảnh chưa qua xử lý
# def detect_red_and_detect_green(img):
#     'img ảnh gốc, trả về image_with_red, ,image_with_green,dilated_mask_red,dilated_mask_green'
#     # Chuyển đổi hình ảnh sang không gian màu HSV
#     image_source = img.copy()
#     hsv = cv.cvtColor(image_source, cv.COLOR_BGR2HSV)
#     image = image_source.copy()
#     # Định nghĩa khoảng giá trị hue (H) cho màu đỏ
#     lower_red1 = np.array([0, 50, 50])
#     upper_red1 = np.array([10, 255, 255])

#     lower_red2 = np.array([170, 50, 50])
#     upper_red2 = np.array([180, 255, 255])

#     lower_green = np.array([40, 40, 40])
#     upper_green = np.array([80, 255, 255])

#     # Tạo mặt nạ để nhận diện màu đỏ
#     mask_red1 = cv.inRange(hsv, lower_red1, upper_red1)
#     mask_red2 = cv.inRange(hsv, lower_red2, upper_red2)
#     mask_green = cv.inRange(hsv, lower_green, upper_green)

#     mask_red = mask_red1 + mask_red2

#     # Giãn vùng màu đỏ
#     dilated_mask_red = dilate_mask(mask_red, iterations=2)
#     dilated_mask_green = dilate_mask(mask_green, iterations=2)

#     # Áp dụng mặt nạ màu đỏ lên ảnh gốc
#     image_with_red = cv.bitwise_and(image_source, image_source, mask=dilated_mask_red)
#     image_with_green = cv.bitwise_and(image, image, mask=dilated_mask_green)

#     return image_with_red,image_with_green,dilated_mask_red,dilated_mask_green
    


# while True:
#     # success, img = cap.read()
#     img = cv.imread('D:/study/NCKH/tensorflow/img_train/Led_background_white/2.jpg')

#     Threshold1 = cv.getTrackbarPos("Threshold1", "Parameters")
#     Threshold2 = cv.getTrackbarPos("Threshold2", "Parameters")
#     Areamin = cv.getTrackbarPos("Areamin", "Parameters")
#     Areamax = cv.getTrackbarPos("Areamax", "Parameters")

#     # tìm màu đỏ và cắt ảnh
#     area_red = 2000
#     area_green = 1000
#     # img_red, img_green, dt_red,dt_green = detect_red_and_detect_green(img)
#     lower_green = np.array([40, 40, 40])
#     upper_green = np.array([80, 255, 255])

#     lower_red1 = np.array([0, 50, 50])
#     upper_red1 = np.array([10, 255, 255])

#     lower_red2 = np.array([170, 50, 50])
#     upper_red2 = np.array([180, 255, 255])

#     lower_black = np.array([0, 0, 0])
#     upper_black = np.array([250, 40, 40])

#     img_gre, dt_gre = detect_one_color(img,lower_green,upper_green,2)
#     img_red, dt_red = detect_two_color(img,lower_red1,upper_red1,lower_red2,upper_red2,2)
#     # img_black, dt_black = detect_one_color(img_red,lower_black,upper_black,2)

#     # black_points = find_black_points(img_red)
#     # print(black_points)
#     imgs = img_gre+img_red
#     img_coutours1 = imgs.copy()
#     img1,gray1,imgCanny1,dilated1,img_cout1,S_none1 = proceed(imgs,img_coutours1,Threshold1,Threshold2,Areamin,Areamax)
#     # gôm ảnh vào một mảng
#     print(S_none1)
#     scale = cv.getTrackbarPos("scale", "Parameters")
#     imgStack = stackImages(scale/10, ([img, dt_red, img_red],
#                                 [imgs, dt_gre, img_gre],
#                                 [img_cout1, imgCanny1, dilated1]))
#     cv.imshow('Result', imgStack)
#     key = cv.waitKey(1)
#     if key == ord('q'):
#         break

# Giải phóng bộ nhớ và đóng cửa sổ
# cap.release()
# cv.destroyAllWindows()


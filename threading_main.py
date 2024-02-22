from threading import Timer,Thread
import threading
import time
from multiprocessing import Value
# global out_all,total_area_green_1,total_area_red_1,count,avg_area_green_1,avg_area_red_1,none_than_thu_1

lock_1 = threading.Lock()
lock_2 = threading.Lock()

total_area_red_1 = 0  # Tổng diện tích của 500 lần xuất ra
total_area_green_1 = 0  # Tổng diện tích của 500 lần xuất ra
avg_area_green_1 = 0  # Giá trị trung bình
avg_area_red_1 = 0  # Giá trị trung bình
none_than_thu_1 = '0'

total_area_red_2 = 10  # Tổng diện tích của 500 lần xuất ra
total_area_green_2 = 10  # Tổng diện tích của 500 lần xuất ra
avg_area_green_2 = 10  # Giá trị trung bình
avg_area_red_2 = 10  # Giá trị trung bình
none_than_thu_2 = '10'

S_min_red_cancel_1 = 3
S_min_green_cancel_1 = 0.3
S_min_red_cancel_2 = 3
S_min_green_cancel_2 = 0.6

S_max_red_cancel_1 = 10
S_max_green_cancel_1 = 2
S_max_red_cancel_2 = 10
S_max_green_cancel_2 = 2

count_1 = 0  # Số lần xuất ra
count_2 = 0  # Số lần xuất ra

out_all_1 = False
out_all_2 = False

time_1 = 0
time_2 = 0

def threading_cam_1():
    import cv2 as cv
    import numpy as np
    from getcoutours import proceed,get_area
    from find_black_red_green import detect_one_color, detect_two_color, detect_color
    from stackImages import stackImages
    import time
    from object_detection import tflite_detect_images
    import time
    import os
    import sys

    # 4:3, 16:9, 3:2
    frameWith1 = 640
    frameHeight1 = 480
    cap = cv.VideoCapture(0)
    cap.set(3, frameWith1)
    cap.set(4, frameHeight1)

    def empty(a):
        pass

    cv.namedWindow("Parameters1")
    cv.resizeWindow("Parameters1", 600, 400)
    cv.createTrackbar("Threshold1", "Parameters1", 20, 255, empty)
    cv.createTrackbar("Threshold2", "Parameters1", 20, 255, empty)
    cv.createTrackbar("L1", "Parameters1", 85, 255, empty)
    cv.createTrackbar("L2", "Parameters1", 104, 255, empty)
    cv.createTrackbar("L3", "Parameters1", 139, 255, empty)
    cv.createTrackbar("H1", "Parameters1", 129, 255, empty)
    cv.createTrackbar("H2", "Parameters1", 121, 255, empty)
    cv.createTrackbar("H3", "Parameters1", 158, 255, empty)
    cv.createTrackbar("scale", "Parameters1", 3, 10, empty)

    # Khởi tạo các biến
    global out_all_1,total_area_green_1,total_area_red_1,count_1,avg_area_green_1,avg_area_red_1,none_than_thu_1

    total_area_red_1 = 0  # Tổng diện tích của 500 lần xuất ra
    total_area_green_1 = 0  # Tổng diện tích của 500 lần xuất ra
    count_1 = 0  # Số lần xuất ra
    avg_area_green_1 = 0  # Giá trị trung bình
    avg_area_red_1 = 0  # Giá trị trung bình
    none_than_thu_1 = '0'
    out_all_1 = False

    if getattr(sys, "frozen", False):
        # Đối với ứng dụng được đóng gói
        folder_path = os.path.dirname(sys.executable)
    else:
        # Đối với mã nguồn Python
        folder_path = os.path.dirname(os.path.abspath(__file__))

    # Đường dẫn tới tệp detect.tflite và labelmap.txt
    PATH_TO_MODEL = os.path.join(folder_path, "detect.tflite")
    PATH_TO_LABELS = os.path.join(folder_path, "labelmap.txt")

    min_conf_threshold1 = 0.97  # Confidence threshold (try changing this to 0.01 if you don't see any detection results)
    images_to_test1 = 10  # Number of images to run detection on

    while True:
        global time_1
        start_time = time.time()
        # out_all = False
        # success, img = cap.read()
        ret, frame_sc = cap.read()
        # Lấy kích thước của ảnh
        height, width, _ = frame_sc.shape

        # Tính vị trí cắt ảnh
        cut_start = int(height / 3)
        cut_end = int(2 * height / 3)

        # Cắt ảnh theo chiều ngang
        frame = frame_sc[cut_start:cut_end:]
        # frame = cv.imread('D:/study/NCKH/tensorflow/img_train/Led_background_white/3.jpg')
        img = frame.copy()
        img_source = frame.copy()
        Threshold1 = cv.getTrackbarPos("Threshold1", "Parameters1")
        Threshold2 = cv.getTrackbarPos("Threshold2", "Parameters1")
        Areamin_red = 3700
        Areamax_red = 30000
        Areamin_green = 370
        Areamax_green = 20000

        L1 = cv.getTrackbarPos("L1", "Parameters1")
        L2 = cv.getTrackbarPos("L2", "Parameters1")
        L3 = cv.getTrackbarPos("L3", "Parameters1")
        H1 = cv.getTrackbarPos("H1", "Parameters1")
        H2 = cv.getTrackbarPos("H2", "Parameters1")
        H3 = cv.getTrackbarPos("H3", "Parameters1")

        color_ranges_greed = [
        # ([L, A, B], [LM, AM, BM]),  # Khoảng màu 1
        # ([58, 116, 133], [148, 122, 153])  # Khoảng màu 2
        ([L1, L2, L3], [H1, H2, H3]) 
        ]
        img_gre, dilated_mask = detect_color(img_source, color_ranges_greed, 2, cv.COLOR_BGR2LAB)

        color_ranges_red = [
        ([0, 75, 75], [10, 255, 255]),  # Khoảng màu 1
        ([160, 75, 75], [180, 255, 255])  # Khoảng màu 2
        ]
        img_red, dilated_mask_red = detect_color(img_source, color_ranges_red, 2, cv.COLOR_BGR2HSV)

        imgs = img_gre + img_red
        img_coutours_red_1 = img_red.copy()
        img_coutours_green_1 = img_gre.copy()

        img_red_1, gray_red_1, imgCanny_red_1, dilated_red_1, img_cout_red_1, S_none_red_1 = proceed(
        img_red, img_coutours_red_1, Threshold1, Threshold2, Areamin_red, Areamax_red
        )
        
        img_green_1, gray_green_1, imgCanny_green_1, dilated_green_1, img_cout_green_1, S_none_green_1 = proceed(
        img_gre, img_coutours_green_1, Threshold1, Threshold2, Areamin_green, Areamax_green
        )
        # print("S_red_1 = ", S_none_red_1)
        # print("S_green_1 = ", S_none_green_1)

        # gôm ảnh vào một mảng
        # Cập nhật tổng diện tích và số lần xuất ra
        S_red_1 = get_area(S_none_red_1, 3700)
        S_green_1 = get_area(S_none_green_1, 3700)

        # print("S_red_1 real = ", S_red_1)
        # print("S_green_1 real = ", S_green_1)

        total_area_red_1 = S_red_1
        total_area_green_1 = S_green_1
        # Run inferencing function!
        result_frame1, detections1, none_than_thu_1 = tflite_detect_images(
            PATH_TO_MODEL,
            frame_sc,
            PATH_TO_LABELS,
            min_conf_threshold1,
            images_to_test1,
            name_error="8",
        )

        scale = cv.getTrackbarPos("scale", "Parameters1")

        imgStack1 = stackImages(
            scale / 10,
            (
                [img_source, imgCanny_red_1, img_cout_red_1],
                [img_green_1, imgCanny_green_1, img_cout_green_1],
                [img_gre, img_red, result_frame1]
            ),
        )
        cv.imshow("Result1", imgStack1)
        with lock_1:
            key1 = cv.waitKey(1)
            # global out_all
            if key1 == ord("q"):
            # if out_all_1 == True:
                # out_all_1 = True
                break
        end_time = time.time()
        time_1 = end_time - start_time
        # print(f"Chương trình 1 chạy hết sau {time_1} giây.")

    # Giải phóng bộ nhớ và đóng cửa sổ
    cap.release()
    # cv.destroyAllWindows()

# Hàm thực hiện công việc cần sau khi luồng 1 chạy
def threading_cam_2():
    import cv2 as cv
    import numpy as np
    from getcoutours import proceed,get_area
    from find_black_red_green import detect_color
    from stackImages import stackImages
    import time
    from object_detection import tflite_detect_images
    import time
    import os
    import sys

    # 4:3, 16:9, 3:2
    frameWith_2 = 640
    frameHeight_2 = 480
    cap_2 = cv.VideoCapture(2)
    cap_2.set(3, frameWith_2)
    cap_2.set(4, frameHeight_2)

    def empty(b):
        pass

    cv.namedWindow("Parameters2")
    cv.resizeWindow("Parameters2", 600, 400)
    cv.createTrackbar("Threshold1", "Parameters2", 20, 255, empty)
    cv.createTrackbar("Threshold2", "Parameters2", 20, 255, empty)
    cv.createTrackbar("L1", "Parameters2", 58, 255, empty)
    cv.createTrackbar("L2", "Parameters2", 111, 255, empty)
    cv.createTrackbar("L3", "Parameters2", 131, 255, empty)
    cv.createTrackbar("H1", "Parameters2", 139, 255, empty)
    cv.createTrackbar("H2", "Parameters2", 119, 255, empty)
    cv.createTrackbar("H3", "Parameters2", 144, 255, empty)
    cv.createTrackbar("scale", "Parameters2", 3, 10, empty)

    # Khởi tạo các biến
    global out_all_2,total_area_green_2,total_area_red_2,count_2,avg_area_green_2,avg_area_red_2,none_than_thu_2

    total_area_red_2 = 0  # Tổng diện tích của 500 lần xuất ra
    total_area_green_2 = 0  # Tổng diện tích của 500 lần xuất ra
    count_2 = 0  # Số lần xuất ra
    avg_area_green_2 = 0  # Giá trị trung bình
    avg_area_red_2 = 0  # Giá trị trung bình
    none_than_thu_2 = '0'
    out_all_2 = False

    if getattr(sys, "frozen", False):
        # Đối với ứng dụng được đóng gói
        folder_path = os.path.dirname(sys.executable)
    else:
        # Đối với mã nguồn Python
        folder_path = os.path.dirname(os.path.abspath(__file__))

    # Đường dẫn tới tệp detect.tflite và labelmap.txt
    PATH_TO_MODEL = os.path.join(folder_path, "detect.tflite")
    PATH_TO_LABELS = os.path.join(folder_path, "labelmap.txt")

    min_conf_threshold2 = 0.97  # Confidence threshold (try changing this to 0.01 if you don't see any detection results)
    images_to_test2 = 10  # Number of images to run detection on

    while True:
        global time_2
        start_time = time.time()
        # out_all_2 = False
        # success, img = cap_2.read()
        ret, frame_sc = cap_2.read()
        # Lấy kích thước của ảnh
        height, width, _ = frame_sc.shape

        # Tính vị trí cắt ảnh
        cut_start = int(height / 3)
        cut_end = int(2* height / 3)

        # Cắt ảnh theo chiều ngang
        frame = frame_sc[cut_start:cut_end:]
        # frame = cv.imread('D:/study/NCKH/tensorflow/img_train/Led_background_white/3.jpg')
        img = frame.copy()
        img_source = frame.copy()
        Threshold1 = cv.getTrackbarPos("Threshold1", "Parameters2")
        Threshold2 = cv.getTrackbarPos("Threshold2", "Parameters2")
        Areamin_red = 4700
        Areamax_red = 40000
        Areamin_green = 470
        Areamax_green = 20000
        L1 = cv.getTrackbarPos("L1", "Parameters2")
        L2 = cv.getTrackbarPos("L2", "Parameters2")
        L3 = cv.getTrackbarPos("L3", "Parameters2")
        H1 = cv.getTrackbarPos("H1", "Parameters2")
        H2 = cv.getTrackbarPos("H2", "Parameters2")
        H3 = cv.getTrackbarPos("H3", "Parameters2")
        color_ranges_greed = [
        # ([L, A, B], [LM, AM, BM]),  # Khoảng màu 1
        ([L1, L2, L3], [H1, H2, H3])  # Khoảng màu 2
        ]
        img_gre, dilated_mask = detect_color(img_source, color_ranges_greed, 2, cv.COLOR_BGR2LAB)

        color_ranges_red = [
        ([0, 75, 75], [10, 255, 255]),  # Khoảng màu 1
        ([160, 75, 75], [180, 255, 255])  # Khoảng màu 2
        ]
        img_red, dilated_mask_red = detect_color(img_source, color_ranges_red, 2, cv.COLOR_BGR2HSV)

        imgs = img_gre + img_red
        img_coutours_red_1 = img_red.copy()
        img_coutours_green_1 = img_gre.copy()

        img_red_1, gray_red_1, imgCanny_red_1, dilated_red_1, img_cout_red_1, S_none_red_1 = proceed(
        img_red, img_coutours_red_1, Threshold1, Threshold2, Areamin_red, Areamax_red
        )

        img_green_1, gray_green_1, imgCanny_green_1, dilated_green_1, img_cout_green_1, S_none_green_1 = proceed(
        img_gre, img_coutours_green_1,
          Threshold1, Threshold2, Areamin_green, Areamax_green
        )
        # print("S_red_2 = ", S_none_red_1)
        # print("S_green_2 = ", S_none_green_1)
        S_red_1 = get_area(S_none_red_1, 4700)
        S_green_1 = get_area(S_none_green_1, 4700)
        # print("S_red_2 real = ", S_red_1)
        # print("S_green_2 real = ", S_green_1)
        # gôm ảnh vào một mảng
        # Cập nhật tổng diện tích và số lần xuất ra
        total_area_red_2 = S_red_1
        total_area_green_2 = S_green_1
        # Run inferencing function!
        result_frame1, detections1, none_than_thu_2 = tflite_detect_images(
            PATH_TO_MODEL,
            frame_sc,
            PATH_TO_LABELS,
            min_conf_threshold2,
            images_to_test2,
            name_error="8",
        )

        scale = cv.getTrackbarPos("scale", "Parameters2")

        imgStack2 = stackImages(
            scale / 10,
            (
                [img_source, imgCanny_red_1, img_cout_red_1],
                [img_green_1, imgCanny_green_1, img_cout_green_1],
                [img_gre, img_red, result_frame1]
            ),
        )
        cv.imshow("Result2", imgStack2)
        with lock_2:
            key2 = cv.waitKey(1)
            # global out_all_2
            if key2 == ord("q"):
            # if out_all_1 == True:
                # out_all_2 = True
                break
        end_time = time.time()
        time_2 = end_time - start_time
        # print(f"Chương trình 2 chạy hết sau {time_2} giây.")

    # Giải phóng bộ nhớ và đóng cửa sổ
    cap_2.release(1)
    # cv.destroyAllWindows()

def get_uart():
    import time
    import serial
    import cv2 as cv
    global matcuon1, gaydoi1, matcuon2, gaydoi2,out_all_1,total_area_green_1,total_area_red_1,count_1,avg_area_green_1,avg_area_red_1,none_than_thu_1,out_all_2,total_area_green_2,total_area_red_2,count_2,avg_area_green_2,avg_area_red_2,none_than_thu_2
            
    matcuon1 = 0
    gaydoi1 = 0
    matcuon2 = 0
    gaydoi2 = 0
    # UART to esp32
    # import matplotlib as plt

    # # ser = serial.Serial('COM5', 9600)  # Thay 'COMx' bằng cổng UART tương ứng trên laptop
    # # time.sleep(2)  # Đợi một chút cho kết nối
    
    # def plot_values(matcuon1, gaydoi1, matcuon2, gaydoi2):
    #     labels = ['Matcuon1', 'Gaydoi1', 'Matcuon2', 'Gaydoi2']
    #     values = [matcuon1, gaydoi1, matcuon2, gaydoi2]

    #     fig, ax = plt.subplots()
    #     bars = ax.bar(labels, values)

    #     # Add labels and title
    #     ax.set_ylabel('Count')
    #     ax.set_title('Count of Matcuon1, Gaydoi1, Matcuon2, Gaydoi2')

    #     # Add values on top of the bars
    #     for bar in bars:
    #         yval = bar.get_height()
    #         ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')

    #     plt.show()
    def output_error():
        import time
        import serial
        import cv2 as cv
        import matplotlib as plt
        from threading import Timer

        global matcuon1, gaydoi1, matcuon2, gaydoi2, time_1,time_2,out_all_1,total_area_green_1,total_area_red_1,count_1,avg_area_green_1,avg_area_red_1,none_than_thu_1,out_all_2,total_area_green_2,total_area_red_2,count_2,avg_area_green_2,avg_area_red_2,none_than_thu_2,S_max_green_cancel_1,S_max_red_cancel_1,S_max_green_cancel_2,S_max_red_cancel_2,S_min_green_cancel_1,S_min_red_cancel_1,S_min_green_cancel_2,S_min_red_cancel_2
        ser = serial.Serial('COM5', 9600)  # Thay 'COMx' bằng cọng UART tương ứng trên laptop
        time.sleep(2)  # Đởi một chút cho kết nối
        avg_area_red_1 = total_area_red_1 
        avg_area_green_1 = total_area_green_1 

        avg_area_red_2 = total_area_red_2 
        avg_area_green_2 = total_area_green_2

        data_alrm = [0, 0, 0, 0, 0, 0, 0, 0]
        if none_than_thu_1 == '8':
            data_alrm[1]=1

        if avg_area_red_1 <= S_min_red_cancel_1 or avg_area_red_1 >= S_max_red_cancel_1:
            data_alrm[2]=20

        if avg_area_green_1 <= S_min_green_cancel_1 or avg_area_green_1 >= S_max_green_cancel_1:
            data_alrm[3]=40

        if none_than_thu_2 == '8':
            data_alrm[5]=2

        if avg_area_red_2 <= S_min_red_cancel_2 or avg_area_red_2 >= S_max_red_cancel_2:
            data_alrm[6]=21

        if avg_area_green_2 <= S_min_green_cancel_2 or avg_area_green_2 >= S_max_green_cancel_2:
            data_alrm[7]=41

        # if  avg_area_green_1 < 0.3 and avg_area_red_1 > 1 :
        #     print("Mất Cuốn 1")
        #     matcuon1 = matcuon1+1
        if avg_area_green_1 > 0.3 and avg_area_red_1 < 3:
            print("Gãy đôi 1")
            gaydoi1 = gaydoi1+1
        if avg_area_green_2 < 0.3 and avg_area_red_2 > 1:
            print("Mất Cuốn 2")
            matcuon2 = matcuon2+1
        if avg_area_green_2 > 0.3 and avg_area_red_2 < 3:
            print("Gãy đôi 2")
            gaydoi2 = gaydoi2+1
        # plot_values(matcuon1, gaydoi1, matcuon2, gaydoi2)
            
        print("S_green_1 = ", avg_area_green_1)
        print("S_red_1 = ", avg_area_red_1)
        print("S_green_2 = ", avg_area_green_2)
        print("S_red_2 = ", avg_area_red_2)
            
        print("Matcuon1 = ", matcuon1)
        print("Gaydoi1 = ", gaydoi1)
        print("Matcuon2 = ", matcuon2)
        print("Gaydoi2 = ", gaydoi2)
            
        # Chuyển đổi mảng thành chuỗi dữ liệu
        data_to_send = ','.join(map(str, data_alrm))
        print("Gửi:", data_to_send)
        # Gửi chuỗi dữ liệu qua UART
        ser.write(data_to_send.encode())  # Chuyển chuỗi thành dạng bytes và gửi

        total_area_red_1 = 1
        total_area_green_1 = 1
        total_area_red_2 = 1
        total_area_green_2 = 1
        none_than_thu_1 = '0'
        none_than_thu_2 = '0'

        my_timer = Timer(1.2, output_error, args=())
        my_timer.start()
        if(out_all_1==True):
            my_timer.cancel()

    output_error()

    while True:
        key = input("Nhấn 'q' để thoát: ")
        if key.lower() == 'q':
            cv.destroyAllWindows()
            out_all_1=True
            break


# Tạo luồng cho việc hiển thị chữ 'a' mỗi 2 giây
thread1 = Thread(target=threading_cam_1)
thread2 = Thread(target=threading_cam_2)
thread3 = Thread(target=get_uart)
# thread4 = Thread(target=update_values_S)
# Bắt đầu chạy các luồng
thread1.start()
thread2.start()
thread3.start()
# thread4.start()

# Chờ các luồng hoàn thành (không cần thiết trong trường hợp này)
thread1.join()
thread2.join()
thread3.join()
# thread4.join()

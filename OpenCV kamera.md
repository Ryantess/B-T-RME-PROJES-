import os
import cv2
from base_camera import BaseCamera
import RPIservo
import numpy as np
import move
import switch
import datetime
import Kalman_filter
import PID
import time
import threading
import imutils
import robotLight

# os: İşletim sistemi ile ilgili işlemler için kullanılır
# cv2: OpenCV kütüphanesi, görüntü işleme ve bilgisayarla görme işlemleri için kullanılır.
# RPIservo: Raspberry Pi servo kontrolü için bir modül.
# numpy (np): Sayısal işlemler ve diziler için kullanılır.
# move: Muhtemelen robotun hareket kontrolü için bir modül.
# switch: Anahtar ve buton kontrolü için bir modül.
# datetime: Tarih ve saat işlemleri için kullanılır.
# Kalman_filter: Kalman filtresi ile ilgili işlemler için bir modül.
# PID: PID kontrolcüsü ile ilgili işlemler için bir modül.
# time: Zaman ile ilgili işlemler için kullanılır.
# threading: Çoklu iş parçacığı oluşturmak ve yönetmek için kullanılır.
# imutils: OpenCV ile görüntü işleme işlemlerini kolaylaştıran yardımcı araçlar sağlar.
# robotLight: Robotun ışık kontrolü için bir modül.

led = robotLight.RobotLight()
pid = PID.PID()
pid.SetKp(0.5)
pid.SetKd(0)
pid.SetKi(0)

CVRun = True
linePos1 = 440
linePos2 = 380
lineColor = 255
frameRender = True
lineErrorMargin = 20
ImgIsNone = False

colorUpper = np.array([44, 255, 255])
colorLower = np.array([24, 100, 100])
# led: Robotun ışık kontrolü için bir nesne.
# pid: PID kontrolcüsü nesnesi ve parametre ayarları.
# CVRun: Görüntü işleme işlemlerinin çalışıp çalışmayacağını kontrol eden bayrak.
# linePos1, linePos2: Çizgi takibi için kullanılan iki yatay çizginin pozisyonları.
# lineColor: Çizgi takibi yapılacak çizginin rengi.
# frameRender: Çerçeve çizim işlemlerini kontrol eden bayrak.
# lineErrorMargin: Çizgi takibinde kabul edilebilir hata payı.
# ImgIsNone: Görüntü boş olup olmadığını kontrol eden bayrak.
# colorUpper, colorLower: Renk takibi için HSV renk aralığı.

class CVThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.kalmanX = Kalman_filter.Kalman_filter(0.01, 0.1)
        self.kalmanY = Kalman_filter.Kalman_filter(0.01, 0.1)
        self.P_servo, self.T_servo = 11, 11
        self.P_angle, self.T_angle = 0, 0
        self.cameraDiagonalW, self.cameraDiagonalH = 64, 48
        self.videoW, self.videoH = 640, 480
        self.Y_lock, self.X_lock = 0, 0
        self.tor = 17

        self.scGear = RPIservo.ServoCtrl()
        self.scGear.moveInit()
        move.setup()
        switch.switchSetup()

        self.CVThreading = False
        self.CVMode = 'none'
        self.imgCV = None

        self.drawing = False
        self.findColorDetection = False

        self.motionCounter = 0
        self.lastMotionCaptured = datetime.datetime.now()

        self.__flag = threading.Event()
        self.__flag.clear()

    # threading.Thread: CVThread sınıfı, threading.Thread sınıfından türetilmiştir ve  çokluiş parçacığı oluşturmayı sağlar.
    # self.font: OpenCV de metin yazmak için kullanılacak font.
    # self.kalmanX, self.kalmanY: X  ve Y eksenlerinde Kalman  filtreleri.
    # self.P_servo, self.T_servo: Pan ve Tilt servo açıları.
    # self.P_angle, self.T_angle: Pan  ve Tilt açıları.
    # self.cameraDiagonalW, self.cameraDiagonalH: Kameranın çapraz genişliği ve yüksekliği.
    # self.videoW, self.videoH: Video  genişliği  ve    yüksekliği.
    # self.Y_lock, self.X_lock: Pan ve Tilt kilit  bayrakları.
    # self.tor: Eşik değer.
    # self.scGear: Servo kontrolcüsü.
    # self.CVThreading: Görüntü işleme  işleminin  devam  edip etmediğini kontrol eden bayrak.
    # self.CVMode: Görüntü işleme modunu tutar.
    # self.imgCV: İşlenecek görüntü.
    # self.drawing: Çizim işleminin aktif olup olmadığını kontrol eden bayrak.
    # self.findColorDetection: Renk takibinin aktif olup olmadığını kontrol eden bayrak
    # self.motionCounter: Hareket sayacını tutar
    # self.lastMotionCaptured: Son hareketin yakalandığı zaman
    # self.__flag: Threadi durdurup devam ettirmek için bir olay bayrağı.

    def run(self):
        while True:
            self.__flag.wait()
            if self.CVMode == 'none':
                continue
            elif self.CVMode == 'findColor':
                self.findColor(self.imgCV)
            elif self.CVMode == 'findlineCV':
                self.findLine(self.imgCV)
            elif self.CVMode == 'watchDog':
                self.watchDog(self.imgCV)
            self.CVThreading = False

    def setMode(self, mode, img):
        self.CVMode = mode
        self.imgCV = img
        self.resume()

    def drawElements(self, img):
        if self.CVMode == 'none':
            pass
        elif self.CVMode == 'findColor':
            status = 'Target Detected' if self.findColorDetection else 'Target Detecting'
            cv2.putText(img, status, (40, 60), self.font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            if self.drawing and self.radius > 10:
                cv2.rectangle(img, (int(self.box_x - self.radius), int(self.box_y + self.radius)),
                              (int(self.box_x + self.radius), int(self.box_y - self.radius)), (255, 255, 255), 1)
        elif self.CVMode == 'findlineCV' and frameRender:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
            img = cv2.erode(img, None, iterations=6)
            try:
                self.drawLineTracking(img)
            except:
                pass
        elif self.CVMode == 'watchDog' and self.drawing:
            cv2.rectangle(img, (self.mov_x, self.mov_y), (self.mov_x + self.mov_w, self.mov_y + self.mov_h), (128, 255, 0), 1)
        return img

    def drawLineTracking(self, img):
        color_msg = 'Following White Line' if lineColor == 255 else 'Following Black Line'
        cv2.putText(img, color_msg, (30, 50), self.font, 0.5, (128, 255, 128), 1, cv2.LINE_AA)

        self.drawVerticalLine(img, self.left_Pos1, linePos1)
        self.drawVerticalLine(img, self.right_Pos1, linePos1)
        self.drawVerticalLine(img, self.left_Pos2, linePos2)
        self.drawVerticalLine(img, self.right_Pos2, linePos2)
        self.drawHorizontalLines(img)

    def drawVerticalLine(self, img, pos, linePos):
        cv2.line(img, (pos, linePos + 30), (pos, linePos - 30), (255, 128, 64), 1)

    def drawHorizontalLines(self, img):
        cv2.line(img, (0, linePos1), (640, linePos1), (255, 255, 64), 1)
        cv2.line(img, (0, linePos2), (640, linePos2), (255, 255, 64), 1)
        center_line = int((linePos1 + linePos2) / 2)
        cv2.line(img, (self.center - 20, center_line), (self.center + 20, center_line), (0, 0, 0), 1)
        cv2.line(img, (self.center, center_line + 20), (self.center, center_line - 20), (0, 0, 0), 1)

    def watchDog(self, img):
        timestamp = datetime.datetime.now()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.avg is None:
            print("[INFO] starting background model...")
            self.avg = gray.copy().astype("float")
            return

        cv2.accumulateWeighted(gray, self.avg, 0.5)
        self.frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(self.avg))
        self.thresh = cv2.threshold(self.frameDelta, 5, 255, cv2.THRESH_BINARY)[1]
        self.thresh = cv2.dilate(self.thresh, None, iterations=2)
        self.cnts = imutils.grab_contours(cv2.findContours(self.thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))

        for c in self.cnts:
            if cv2.contourArea(c) < 5000:
                continue
            (self.mov_x, self.mov_y, self.mov_w, self.mov_h) = cv2.boundingRect(c)
            self.drawing = True
            self.motionCounter += 1
            self.lastMotionCaptured = timestamp
            led.setColor(255, 78, 0)

        if (timestamp - self.lastMotionCaptured).seconds >= 0.5:
            led.setColor(0, 78, 255)
            self.drawing = False
        self.pause()

    def findLine(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, threshold_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU)
        eroded_img = cv2.erode(threshold_img, None, iterations=6)
        self.processLine(eroded_img, linePos1)
        self.processLine(eroded_img, linePos2)
        self.center = int((self.center_Pos1 + self.center_Pos2) / 2)
        self.findLineCtrl(self.center, 320)
        self.pause()

    def processLine(self, img, linePos):
        line_color_count = np.sum(img[linePos] == lineColor)
        line_indexes = np.where(img[linePos] == lineColor)
        if line_color_count == 0:
            line_color_count = 1

        left_pos = line_indexes[0][line_color_count - 1]
        right_pos = line_indexes[0][0]
        center_pos = int((left_pos + right_pos) / 2)

        if linePos == linePos1:
            self.left_Pos1, self.right_Pos1, self.center_Pos1 = left_pos, right_pos, center_pos
        else:
            self.left_Pos2, self.right_Pos2, self.center_Pos2 = left_pos, right_pos, center_pos

    def findLineCtrl(self, pos, center):
        error = (pos - center) / 3
        out_val = int(round(pid.GenOut(error), 0))
        if pos > (center + lineErrorMargin):
            self.moveAndAdjustServo(0, -out_val, 'right')
        elif pos < (center - lineErrorMargin):
            self.moveAndAdjustServo(0, out_val, 'left')
        else:
            move.move(80, 'forward', 'no', 0.5)

    def moveAndAdjustServo(self, servo, out_val, direction):
        self.scGear.moveAngle(servo, out_val)
        move.move(80, 'no', direction, 0.5

)

    def findColor(self, img):
        blurred = cv2.GaussianBlur(img, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, colorLower, colorUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            self.box_x, self.box_y = self.kalmanX.kalman(x), self.kalmanY.kalman(y)
            self.radius = radius
            self.drawing = True

            if radius > 10:
                self.P_angle = self.calculateAngle(self.box_x, self.videoW, self.cameraDiagonalW)
                self.T_angle = self.calculateAngle(self.box_y, self.videoH, self.cameraDiagonalH)

                if not self.Y_lock:
                    self.T_servo = self.clampAngle(Servo_dir[2] + self.T_angle)
                if not self.X_lock:
                    self.P_servo = self.clampAngle(Servo_dir[3] - self.P_angle)

                self.scGear.moveAngle(2, self.T_servo)
                self.scGear.moveAngle(3, self.P_servo)
                move.move(80, 'forward', 'no', 0.5)
            else:
                move.move(80, 'no', 'no', 0.5)
            self.findColorDetection = True
        else:
            self.findColorDetection = False
            move.move(80, 'no', 'no', 0.5)
        self.pause()

    def calculateAngle(self, pos, video_dim, camera_dim):
        return -(pos - (video_dim / 2)) / (video_dim / camera_dim)

    def clampAngle(self, angle):
        return max(min(angle, 140), 40)

    def pause(self):
        self.__flag.clear()

    def resume(self):
        self.__flag.set()

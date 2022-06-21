import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtWidgets import QWidget, QApplication, QLabel
from PyQt5.QtCore import QPoint, Qt, QMimeData
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QMouseEvent, QDrag, QPixmap, QImage
import cv2
import numpy as np

#UI파일 연결
#단, UI파일은 Python 코드 파일과 같은 디렉토리에 위치해야한다.
form_class = uic.loadUiType("untitled.ui")[0]

class Button(QLabel):
    def __init__(self, parent):
        QLabel.__init__(self, parent)

    def mouseMoveEvent(self, e: QMouseEvent):
        # 왼쪽 버튼은 클릭용이므로 오른쪽 버튼 입력 허용
        if e.buttons() != Qt.RightButton:
            return
        # 데이터 전송을 위한 MIME 객체 선언
        # 데이터 타입, 보낼 데이터를 Bytes 형으로 저장
        mime_data = QMimeData()
        mime_data.setData("application/hotspot", b"%d %d" % (e.x(), e.y()))

        drag = QDrag(self)
        # MIME 타입데이터를 Drag에 설정
        drag.setMimeData(mime_data)
        # 드래그시 위젯의 모양 유지를 위해 QPixmap에 모양을 렌더링
        pixmap = QPixmap(self.size())
        self.render(pixmap)
        drag.setPixmap(pixmap)

        drag.setHotSpot(e.pos() - self.rect().topLeft())
        drag.exec_(Qt.MoveAction)

#화면을 띄우는데 사용되는 Class 선언
class WindowClass(QMainWindow, form_class):
    def __init__(self) :
        super().__init__()
        self.setupUi(self)
        self.label.setPixmap(QPixmap("bg/bg1.jpg"))
        self.setAcceptDrops(True)
        self.chromakey()

    def chromakey(self):
        img1 = cv2.imread("e.jpg")
        img2 = cv2.imread("bg1.jpg")
        height1, width1 = img1.shape[:2]
        height2, width2 = img2.shape[:2]
        # 고정 위치 (센터)
        x = (width2 - width1) // 2
        y = height2 - height1
        w = x + width1
        h = y + height1
        # --③ 크로마키 배경 영상에서 크로마키 영역을 10픽셀 정도로 지정
        chromakey = img1[:10, :10, :]
        offset = 20

        # --④ 크로마키 영역과 영상 전체를 HSV로 변경

        hsv_chroma = cv2.cvtColor(chromakey, cv2.COLOR_BGR2HSV)
        hsv_img = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)

        # --⑤ 크로마키 영역의 H값에서 offset 만큼 여유를 두어서 범위 지정
        # offset 값은 여러차례 시도 후 결정
        # chroma_h = hsv_chroma[0]
        chroma_h = hsv_chroma[:, :, 0]
        lower = np.array([chroma_h.min() - offset, 100, 100])
        upper = np.array([chroma_h.max() + offset, 255, 255])

        # --⑥ 마스크 생성 및 마스킹 후 합성
        mask = cv2.inRange(hsv_img, lower, upper)
        mask_inv = cv2.bitwise_not(mask)
        roi = img2[y:h, x:w]  # 고정 위치
        fg = cv2.bitwise_and(img1, img1, mask=mask_inv)
        h, w, c = img1.shape
        qImg = QImage(fg, w, h, w*c, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qImg)
        #pixmap = QPixmap('fg/a.jpg')
        self.btn = Button(self)
        self.btn.setPixmap(pixmap)
        # self.btn.setFixedSize(500, 500)
        self.btn.show()

    def dragEnterEvent(self, e: QDragEnterEvent):
        e.accept()

    def dropEvent(self, e: QDropEvent):
        position = e.pos()

        # 보내온 데이터를 받기
        # 그랩 당시의 마우스 위치값을 함께 계산하여 위젯 위치 보정
        offset = e. mimeData().data("application/hotspot")
        x, y = offset.data().decode('utf-8').split()
        self.btn.move(position - QPoint(int(x), int(y)))

        e.setDropAction(Qt.MoveAction)
        e.accept()



if __name__ == "__main__" :
    #QApplication : 프로그램을 실행시켜주는 클래스
    app = QApplication(sys.argv)

    #WindowClass의 인스턴스 생성
    myWindow = WindowClass()

    #프로그램 화면을 보여주는 코드
    myWindow.show()

    #프로그램을 이벤트루프로 진입시키는(프로그램을 작동시키는) 코드
    app.exec_()
from rich.traceback import install
from rich.console import Console
from rich.progress import track
import numpy as np
import random
import click
import cv2
import os

imgPath = 'images'
lblPath = 'labels'

BANNER = """
[bold blue]=====================================================================
        ░██████╗██╗░░██╗██╗░░░██╗░██████╗██╗░░░██╗░██████╗
        ██╔════╝██║░██╔╝╚██╗░██╔╝██╔════╝╚██╗░██╔╝██╔════╝
        ╚█████╗░█████═╝░░╚████╔╝░╚█████╗░░╚████╔╝░╚█████╗░
        ░╚═══██╗██╔═██╗░░░╚██╔╝░░░╚═══██╗░░╚██╔╝░░░╚═══██╗
        ██████╔╝██║░╚██╗░░░██║░░░██████╔╝░░░██║░░░██████╔╝
        ╚═════╝░╚═╝░░╚═╝░░░╚═╝░░░╚═════╝░░░░╚═╝░░░╚═════╝░

          		        [italic]Chroma key TOOL (skysys) V0.1[/italic]
=====================================================================[/bold blue]                                                                                                                                
"""
console = Console()
install()


@click.group()
def main():
    """
    Image Dataset Builder CLI to create amazing datasets
    """
    pass


def create_chromakey(images_path, background_path, location_type, class_num, dataset_type):
    # img 파일만 가져오기
    images_list = os.listdir(images_path)
    file_list_img = [file for file in images_list if file.endswith(".jpg")]
    background_list = os.listdir(background_path)
    file_list_bg = [file for file in background_list if file.endswith(".jpg")]
    for n in track(range(len(file_list_img)), description="Processing..."):
        for file in file_list_bg:
            # --① 크로마키 배경 영상과 합성할 배경 영상 읽기
            filename = file.rstrip('.jpg')

            image = cv2.imread(os.path.abspath(images_path) + "/" + file_list_img[n])
            image_gray = cv2.imread(os.path.abspath(images_path) + "/" + file_list_img[n], cv2.IMREAD_GRAYSCALE)

            blur = cv2.GaussianBlur(image_gray, ksize=(5, 5), sigmaX=0)

            """
            cv2.Canny(gray_img, threshold1, threshold2)
            - threshold1 : 다른 엣지와의 인접 부분(엣지가 되기 쉬운 부분)에 있어 엣지인지 아닌지를 판단하는 임계값
            - threshold2 : 엣지인지 아닌지를 판단하는 임계값

            외곽선(엣지) 검출 파라미터 조정을 하는 방법
            1. 먼저 threshold1와 threshold2를 같은 값으로 한다.
            2. 검출되길 바라는 부분에 엣지가 표시되는지 확인하면서 threshold2 값을 조정한다.
            3. 2번의 조정이 끝나면, threshold1를 사용하여 엣지를 연결시킨다.
            """
            edged = cv2.Canny(blur, 10, 250)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            contours_xy = np.array(contours)
            contours_xy.shape

            # x의 min과 max 찾기
            x_min, x_max = 0, 0
            value = list()
            for i in range(len(contours_xy)):
                for j in range(len(contours_xy[i])):
                    value.append(contours_xy[i][j][0][0])  # 네번째 괄호가 0일때 x의 값
                    x_min = min(value)
                    x_max = max(value)

            # y의 min과 max 찾기
            y_min, y_max = 0, 0
            value = list()
            for i in range(len(contours_xy)):
                for j in range(len(contours_xy[i])):
                    value.append(contours_xy[i][j][0][1])  # 네번째 괄호가 0일때 x의 값
                    y_min = min(value)
                    y_max = max(value)

            # image trim 하기
            x = x_min
            y = y_min
            w = x_max - x_min
            h = y_max - y_min

            img_trim = image[y:y + h, x:x + w]
            cv2.imwrite(os.path.abspath(images_path) + "/" + file_list_img[n], img_trim)

            img1 = cv2.imread(os.path.abspath(images_path) + "/" + file_list_img[n])
            img2 = cv2.imread(os.path.abspath(background_path) + "/" + file)

            # --② ROI 선택을 위한 좌표 계산
            height1, width1 = img1.shape[:2]
            height2, width2 = img2.shape[:2]
            if location_type == 1:
                x_end = width2 - width1
                y_end = height2 - height1
                # 랜덤 위치
                x = random.randrange(0, x_end)
                y = random.randrange(600, y_end)
                w = x + width1
                h = y + height1
            else:
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
            roi = img2[y:h, x:w]
            fg = cv2.bitwise_and(img1, img1, mask=mask_inv)
            bg = cv2.bitwise_and(roi, roi, mask=mask)
            img2[y:h, x:w] = fg + bg

            # --⑦ 결과 출력
            # cv2.imshow('chromakey', img1)
            # cv2.imshow('added', img2)
            createDirectory('{}'.format(imgPath))
            createDirectory('{}'.format(lblPath))
            if dataset_type == 1:
                f = open("labels/{}_{}.txt".format(filename, n), 'w')
                b = (x, w, y, h)
                bb = convert((width2, height2), b)
                f.write("{} {} {} {} {}".format(class_num, bb[0], bb[1], bb[2], bb[3]))
                f.close()
            elif dataset_type == 2:
                f = open("labels/{}_{}.txt".format(filename, n), 'w')
            else:
                f = open("labels/{}_{}.txt".format(filename, n), 'w')

            cv2.imwrite("./images/{}_{}.jpg".format(filename, n), img2)
            cv2.waitKey()
            cv2.destroyAllWindows()
            # console.print()

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = round(x * dw, 6)
    w = round(w * dw, 6)
    y = round(y * dh, 6)  # 6자리 표시
    h = round(h * dh, 6)
    return x, y, w, h

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        console.print("[bold red]Error: Failed to create the directory.[/bold red]")


@main.command()
def version():
    """
    Shows what version idt is currently on
    """
    console.print("[bold]Chroma key Tool (skysys)[/bold] version 0.1")


@main.command()
def authors():
    """
    Shows who are the creators of IDT
    """
    console.print(
        "[bold]Chroma key Tool (skysys)[/bold] email [bold magenta]skysys@skysys.co.kr[/bold magenta] tel [bold red]052) 707-7561 [/bold red]")


@main.command()
@click.option('--default', '-d', is_flag=True, default=False, help="Generate a default config file")
def init(default):
    """
    This command initialyzes idt and creates a dataset config file
    """
    console.clear()
    console.print(BANNER)

    if default:
        document_dict = {
            "images_path": "fg",
            "background_path": "bg",
            "type": 1,
            "IMAGE_SIZE": 512,
            "CLASSES": [{"CLASS_NAME": "Test", "SEARCH_KEYWORDS": "images of cats"}]
        }
        create_chromakey(document_dict["images_path"], document_dict["background_path"], document_dict["type"])

    else:
        while True:
            images_path = click.prompt("Insert your Chroma key images path")
            try:
                file_list = os.listdir(images_path)
                file_list_img = [file for file in file_list if file.endswith(".jpg")]
                if not file_list_img:
                    raise
            except FileNotFoundError:
                console.print("[bold red]Error: Invalid images file path.[/bold red]")
            except:
                console.print("[bold red]Error: chromakey images file does not exist.[/bold red]")
            else:
                break

        console.clear()
        console.print(BANNER)
        while True:
            background_path = click.prompt("Insert your background images path")
            try:
                file_list = os.listdir(background_path)
                file_list_img = [file for file in file_list if file.endswith(".jpg")]
                if not file_list_img:
                    raise
            except FileNotFoundError:
                console.print("[bold red]Error: Invalid background images path.[/bold red]")
            except:
                console.print("[bold red]Error: background images file does not exist.[/bold red]")
            else:
                break

        console.clear()
        console.print(BANNER)
        console.print("[bold]Choose image position[/bold]", justify="left")
        console.print("""
        [1] random [bold blue](recommended)[/bold blue]
        [2] center
        """)
        location_type = click.prompt("What image location do you want?", type=int)

        console.clear()
        console.print(BANNER)
        console.print("[bold]Enter the desired class number [/bold][bold blue](start : 0)[/bold blue]")
        class_num = click.prompt("Insert class number", type=int)

        console.clear()
        console.print(BANNER)
        console.print("[bold]Choose dataset format[/bold]")
        console.print("""
        [1] YOLO 
        [2] COCO
        [3] PASCAL VOC
        """)
        dataset_type = click.prompt("Insert dataset format", type=int)

        # console.clear()
        # console.print(BANNER)
        # console.print("[bold]Choose image resolution[/bold]", justify="left")
        # console.print("""
        # [1] 512 pixels / 512 pixels [bold blue](recommended)[/bold blue]
        # [2] 1024 pixels / 1024 pixels
        # [3] 256 pixels / 256 pixels
        # [4] 128 pixels / 128 pixels
        # [5] Keep original image size
        #
        # [italic]ps: note that the aspect ratio of the image will [bold]not[/bold] be changed, so possibly the images received will have slightly different size[/italic]
        # """)
        #
        #     image_size_ratio = click.prompt("What is the desired image size ratio", type=int)
        #     while image_size_ratio < 1 or image_size_ratio > 5:
        #         console.print("[italic red]Invalid option, please choose between 1 and 5. [/italic red]")
        #         image_size_ratio = click.prompt("\nOption: ", type=int)
        #
        #     if image_size_ratio == 1:
        #         image_size_ratio = 512
        #     elif image_size_ratio == 2:
        #         image_size_ratio = 1024
        #     elif image_size_ratio == 3:
        #         image_size_ratio = 256
        #     elif image_size_ratio == 4:
        #         image_size_ratio = 128
        #     elif image_size_ratio == 5:
        #         image_size_ratio = 0

        # console.clear()
        create_chromakey(images_path, background_path, location_type, class_num, dataset_type)


if __name__ == "__main__":
    main()
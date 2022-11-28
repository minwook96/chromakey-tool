from rich.traceback import install
from rich.console import Console
from rich.progress import track
import numpy as np
import darknet
import random
import click
import cv2
import os

imgPath = 'output'

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


def create_chromakey(foreground_path, background_path, location_type, image_saturation, image_illuminance, image_size,
                     dataset_type, auto_labeling):
    network, class_names, class_colors = "", "", ""
    if auto_labeling:
        random.seed(3)  # deterministic bbox colors
        network, class_names, class_colors = darknet.load_network(
            "./cfg/yolov4.cfg",
            "./cfg/custom.data",
            "./weights/yolov4_object_best.weights",
            batch_size=1
        )

    # img 파일만 가져오기
    images_list = os.listdir(foreground_path)
    file_list_img = [file for file in images_list if
                     file.endswith(".jpg") | file.endswith(".JPG") | file.endswith(".png")]
    background_list = os.listdir(background_path)
    file_list_bg = [file for file in background_list if
                    file.endswith(".jpg") | file.endswith(".JPG") | file.endswith(".png")]
    create_directory('{}'.format(imgPath))

    for n in track(range(len(file_list_img)), description="Processing..."):
        for file in file_list_bg:
            # --① 크로마 키 배경 영상과 합성할 배경 영상 읽기
            filename = file.rstrip('.jpg')

            img1 = cv2.imread(os.path.abspath(foreground_path) + "/" + file_list_img[n])
            img2 = cv2.imread(os.path.abspath(background_path) + "/" + file)
            img1 = cv2.resize(img1, (1728, 972))
            # --② ROI 선택을 위한 좌표 계산
            height1, width1 = img1.shape[:2]
            height2, width2 = img2.shape[:2]
            if location_type == 1:
                x_end = width2 - width1
                y_end = height2 - height1
                # fg가 해상도가 더 높으면 오류 발생 -> 소스 코드 수정 해야함
                # 랜덤 위치
                x = random.randrange(0, x_end)
                # y = random.randrange(0, y_end) #y_end - 50
                y = y_end
                w = x + width1
                h = y + height1
            else:
                # 고정 위치 (센터)
                x = (width2 - width1) // 2
                y = height2 - height1
                w = x + width1
                h = y + height1

            # --③ 크로마 키 배경 영상에서 크로마 키 영역을 10픽셀 정도로 지정
            chromakey = img1[:10, :10, :]
            offset = 20

            # --④ 크로마 키 영역과 영상 전체를 HSV로 변경
            hsv_chroma = cv2.cvtColor(chromakey, cv2.COLOR_BGR2HSV)
            hsv_img = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)

            # --⑤ 크로마 키 영역의 H값에서 offset 만큼 여유를 두어서 범위 지정
            # offset 값은 여러 차례 시도 후 결정
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

            if dataset_type == 1:
                f = open("{}/{}_{}.txt".format(imgPath, filename, n), 'w')
                # b = (x, w, y, h)
                # bb = convert((width2, height2), b)
                # f.write("{} {} {} {} {}\n".format("0", bb[0], bb[1], bb[2], bb[3]))
                # f.close()
            elif dataset_type == 2:
                f = open("{}/{}_{}.txt".format(imgPath, filename, n), 'w')
            else:
                f = open("{}/{}_{}.txt".format(imgPath, filename, n), 'w')

            # 이미지 채도 조정
            if image_saturation > 0:
                img2 = increase_saturation(img2, image_saturation)

            # 이미지 조도 조정
            if image_illuminance > 0:
                img2 = increase_illuminance(img2, image_illuminance)

            # 이미지 크기 조정
            img2 = image_resize(img2, image_size)

            cv2.imwrite("./{}/{}_{}.jpg".format(imgPath, filename, n), img2)
            if auto_labeling:
                detection_image = "./{}/{}_{}.jpg".format(imgPath, filename, n)
                image, detections = image_detection(detection_image, network, class_names, class_colors, .25)
                save_annotations(detection_image, image, detections, class_names)
            f.close()
            cv2.waitKey()
            cv2.destroyAllWindows()
            # console.print()


# 이미지 채도 조정
def increase_saturation(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    s[s > lim] = 255
    s[s <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    return img


# 이미지 조도 조정
def increase_illuminance(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    return img


# 이미지 크기 조정
def image_resize(img, value):
    if value != 4:
        resize_img = cv2.resize(img, (value, value), interpolation=cv2.INTER_AREA)
    else:
        resize_img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)
    return resize_img


# Yolo 형식으로 좌표 변환
def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = round(x * dw, 6)
    w = round(w * dw, 6)
    y = round(y * dh, 6)  # 6자리 표시
    h = round(h * dh, 6)
    return x, y, w, h


def create_directory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        console.print("[bold red]Error: Failed to create the directory.[/bold red]")


def image_detection(image_or_path, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image = cv2.imread(image_or_path)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections


def convert2relative(image, bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = bbox
    height, width, _ = image.shape
    return x / width, y / height, w / width, h / height


def save_annotations(name, image, detections, class_names):
    """
    Files saved with image_name.txt and relative coordinates
    """
    file_name = os.path.splitext(name)[0] + ".txt"
    # print(detections)
    with open(file_name, "w+") as f:
        for label, confidence, bbox in detections:
            x, y, w, h = convert2relative(image, bbox)
            label = class_names.index(label)
            f.write("{} {:.f} {:.f} {:.f} {:.f}\n".format(label, x, y, w, h))
        f.close()


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
        "[bold]Chroma key Tool (skysys)[/bold] email [bold magenta]skysys@skysys.co.kr[/bold magenta] tel [bold "
        "red]052) 707-7561 [/bold red]")


@main.command()
def detection():
    while True:  # 크로마 키 이미지 경로 입력
        images_path = click.prompt("Insert your detection images path")
        try:
            file_list = os.listdir(images_path)
            file_list_img = [file for file in file_list if
                             file.endswith(".jpg") | file.endswith(".JPG") | file.endswith(".png")]
            if not file_list_img:
                raise
        except FileNotFoundError:
            console.print("[bold red]Error: Invalid images file path.[/bold red]")
        else:
            break

    console.clear()
    console.print(BANNER)
    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        "./cfg/yolov4-p6.cfg",
        "./cfg/coco.data",
        "./weights/yolov4-p6.weights",
        batch_size=1
    )
    images_list = os.listdir(images_path)
    file_list_img = [file for file in images_list if
                     file.endswith(".jpg") | file.endswith(".JPG") | file.endswith(".png")]
    for n in track(range(len(file_list_img)), description="Processing..."):
        filename = file_list_img[n].rstrip('.jpg')
        print(file_list_img[n])
        f = open('{}/{}.txt'.format(images_path, filename), "w")
        # list = f.readline()
        # print(list)
        # --① 크로마 키 배경 영상과 합성할 배경 영상 읽기
        # image = cv2.imread(os.path.abspath(images_path) + "/" + file_list_img[n])
        detection_image = "./{}/{}".format(images_path, file_list_img[n])
        image, detections = image_detection(detection_image, network, class_names, class_colors, .25)
        for label, confidence, bbox in detections:
            x, y, w, h = convert2relative(image, bbox)
            label = class_names.index(label)
            if label == 0:
                # print(label, x, y, w, h)
                f.write("{} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(label, x, y, w, h))
        print("----------------------------------")
        f.close()
        # if detections[0][0] == 'worker':
        # x, y, w, h = convert2relative(image, bbox)
        # label = class_names.index(label)
        # f.write("{} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(label, x, y, w, h))
        # save_annotations(detection_image, image, detections, class_names)
        # console.print()



@main.command()
def line():
    while True:  # 크로마 키 이미지 경로 입력
        foreground_path = click.prompt("Insert your Chroma key images path")
        try:
            file_list = os.listdir(foreground_path)
            file_list_img = [file for file in file_list if
                             file.endswith(".jpg") | file.endswith(".JPG") | file.endswith(".png")]
            if not file_list_img:
                raise
        except FileNotFoundError:
            console.print("[bold red]Error: Invalid images file path.[/bold red]")
        else:
            break

    console.clear()
    console.print(BANNER)
    images_list = os.listdir(foreground_path)
    file_list_img = [file for file in images_list if
                     file.endswith(".jpg") | file.endswith(".JPG") | file.endswith(".png")]
    create_directory('fg_line')

    for n in track(range(len(file_list_img)), description="Processing..."):
        # --① 크로마 키 배경 영상과 합성할 배경 영상 읽기
        image = cv2.imread(os.path.abspath(foreground_path) + "/" + file_list_img[n])
        image_gray = cv2.imread(os.path.abspath(foreground_path) + "/" + file_list_img[n], cv2.IMREAD_GRAYSCALE)

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

        # x의 min, max 찾기
        x_min, x_max = 0, 0
        value = list()
        for i in range(len(contours_xy)):
            for j in range(len(contours_xy[i])):
                value.append(contours_xy[i][j][0][0])  # 네번째 괄호가 0일때 x의 값
                x_min = min(value)
                x_max = max(value)

        # y의 min, max 찾기
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
        cv2.imwrite("fg_line/" + file_list_img[n], img_trim)


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
            "foreground_path": "fg",
            "background_path": "bg",
            "location_type": 0,
            "image_saturation": 10,
            "image_illuminance": 50,
            "image_size": 608,
            "dataset_type": 1,
            "auto_labeling": True
        }
        create_chromakey(document_dict["foreground_path"], document_dict["background_path"],
                         document_dict["location_type"], document_dict["image_saturation"],
                         document_dict["image_illuminance"], document_dict["image_size"], document_dict["dataset_type"],
                         document_dict["auto_labeling"])
    else:
        # 크로마 키 이미지 경로 입력
        while True:
            foreground_path = click.prompt("Insert your Chroma key images path")
            try:
                file_list = os.listdir(foreground_path)
                file_list_img = [file for file in file_list if
                                 file.endswith(".jpg") | file.endswith(".JPG") | file.endswith(".png")]
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

        # 백그라운드 이미지 경로 입력
        while True:
            background_path = click.prompt("Insert your background images path")
            try:
                file_list = os.listdir(background_path)
                file_list_img = [file for file in file_list if file.endswith(".jpg") | file.endswith(".png")]
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

        # 합성 시 크로마 키 이미지 위치 정하기
        while True:
            console.print("[bold]Choose image position[/bold]", justify="left")
            console.print("""
                    [1] random [bold blue](recommended)[/bold blue]
                    [2] center
                    """)
            location_type = click.prompt("What image location do you want?", type=int)
            if location_type == 1 or location_type == 2:
                break
            else:
                console.print("[bold red]Error: Invalid location type.[/bold red]")

        # 이미지 채도 지정
        console.clear()
        console.print(BANNER)
        console.print("[bold]Choose image saturation format[/bold]")
        console.print("""
                0(origin) ~ 255
                """)
        image_saturation = click.prompt("Insert saturation", type=int)

        # 이미지 조도 지정
        console.clear()
        console.print(BANNER)
        console.print("[bold]Choose image illuminance format[/bold]")
        console.print("""
                0(origin) ~ 255
                """)
        image_illuminance = click.prompt("Insert illuminance", type=int)

        # 이미지 사이즈 지정
        console.clear()
        console.print(BANNER)
        console.print("[bold]Choose image size format[/bold]")
        console.print("""
                [1] 320x320
                [2] 416x416
                [3] 608x608
                [4] 1920x1080
                """)
        image_size = click.prompt("Insert image size", type=int)
        if image_size == 1:
            image_size = 320
        elif image_size == 2:
            image_size = 416
        elif image_size == 3:
            image_size = 608
        else:
            image_size = 4

        # 데이터셋 형태 지정
        console.clear()
        console.print(BANNER)
        console.print("[bold]Choose dataset format[/bold]")
        console.print("""
        [1] YOLO 
        [2] COCO [bold red]undeveloped[/bold red]
        [3] PASCAL VOC [bold red]undeveloped[/bold red]
        """)
        dataset_type = click.prompt("Insert dataset format", type=int)

        # auto labeling 선택
        console.clear()
        console.print(BANNER)
        console.print("[bold]Choose auto labeling[/bold]")
        console.print("""
        [1] YES 
        [2] NO
        """)
        auto_labeling = click.prompt("Insert ", type=int)
        if auto_labeling == 1:
            auto_labeling = True
        else:
            auto_labeling = False

        create_chromakey(foreground_path, background_path, location_type, image_saturation, image_illuminance,
                         image_size, dataset_type, auto_labeling)


if __name__ == "__main__":
    main()

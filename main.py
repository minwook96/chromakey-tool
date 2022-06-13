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
          
          		        [italic]Chromakey TOOL (skysys) V0.1[/italic]

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

def create_chromakey(images_path, background_path, type):
    # img 파일만 가져오기
    images_list = os.listdir(images_path)
    file_list_img = [file for file in images_list if file.endswith(".jpg")]
    background_list = os.listdir(background_path)
    file_list_bg = [file for file in background_list if file.endswith(".jpg")]
    for n in track(range(len(file_list_img)), description="Processing..."):
        for file in file_list_bg:
            # --① 크로마키 배경 영상과 합성할 배경 영상 읽기
            filename = file.rstrip('.jpg')
            img1 = cv2.imread(os.path.abspath(images_path) + "/" + file_list_img[n])
            img2 = cv2.imread(os.path.abspath(background_path) + "/" + file)

            # --② ROI 선택을 위한 좌표 계산
            height1, width1 = img1.shape[:2]
            height2, width2 = img2.shape[:2]
            if type == 1:
                x = width2 - width1
                y = height2 - height1
                # 랜덤 위치
                rand_x = random.randrange(0, x)
                rand_y = random.randrange(0, y)
                rand_w = rand_x + width1
                rand_h = rand_y + height1
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
            if type == 1:
                roi = img2[rand_y:rand_h, rand_x:rand_w]  # 랜덤 위치
            else:
                roi = img2[y:h, x:w]  # 고정 위치
            fg = cv2.bitwise_and(img1, img1, mask=mask_inv)
            bg = cv2.bitwise_and(roi, roi, mask=mask)
            if type == 1:
                img2[rand_y:rand_h, rand_x:rand_w] = fg + bg  # 랜덤 위치
            else:
                img2[y:h, x:w] = fg + bg  # 고정 위치

            # --⑦ 결과 출력
            # cv2.imshow('chromakey', img1)
            # cv2.imshow('added', img2)
            createDirectory('{}'.format(imgPath))
            createDirectory('{}'.format(lblPath))
            cv2.imwrite("./images/{}_{}.jpg".format(filename, n), img2)
            cv2.waitKey()
            cv2.destroyAllWindows()
            # console.print()

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
    click.clear()
    console.print("[bold magenta]Chromakey Tool (skysys)[/bold magenta] version 0.1")

@main.command()
def authors():
    """
    Shows who are the creators of IDT
    """
    click.clear()
    console.print(
        "[bold]Chromakey Tool (skysys)[/bold] email [bold magenta]skysys@skysys.co.kr[/bold magenta] tel [bold red]052) 707-7561 [/bold red]")

@main.command()
@click.option('--default', '-d', '--d', is_flag=True, default=False, help="Generate a default config file")
def init(default):
    """
    This command initialyzes idt and creates a dataset config file
    """
    console.clear()
    console.print(BANNER)

    if default:
        document_dict = {
            "images_path": "images",
            "background_path": "background",
            "dataset": 50,
            "IMAGE_SIZE": 512,
            "RESIZE_METHOD": "longer_side",
            "CLASSES": [{"CLASS_NAME": "Test", "SEARCH_KEYWORDS": "images of cats"}]}

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
    type = click.prompt("What image location do you want?", type=int)

    # console.clear()
    # console.print(BANNER)
    # dataset = click.prompt("Number of datasets to create per Chroma Key image: ", type=int)

    #     console.clear()
    #     console.print(BANNER)
    #     console.print("[bold]Choose image resolution[/bold]", justify="left")
    #     console.print("""
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
    create_chromakey(images_path, background_path, type)


if __name__ == "__main__":
    main()
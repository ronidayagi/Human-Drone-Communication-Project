import numpy as np
import pyrealsense2 as rs
import cv2
import time
from tkinter import *
from tkinter import ttk

root = Tk()
# x0_in_camera = 0
# y0_in_camera = 0
# x0_in_UI = 320
y0_in_ui = 240
x = 640
y = 420
iteration_in = [0,0,0,0,0,0]
font_in = [36,36,36,36,24,200]
how_mach_make_to_font_bigger = 5

Rsel = 0

headline = None
poligon = None
rects = []
answs = []
Questionair = []
cur_q = 0

QT_2ANS = 1
QT_4ANS = 2
QT_RULLER = 3

class Question:
    _text = ''
    _type = QT_2ANS
    _answers = []
    _choice = -1
    def __init__(self, *args:"text, type, answers"):
        if len(args) > 0: self._text = args[0]
        if len(args) > 1: self._type = args[1]
        if len(args) > 2: self._answers = args[2]

ans1=Questionair.append(Question("Are you hurt?", QT_2ANS, ['Yes', 'No']));
Questionair.append(Question("Grade your happiness", QT_RULLER))
Questionair.append(Question("are you happy?", QT_2ANS, ['Yes', 'No']))
Questionair.append(Question("Choose a number", QT_4ANS, ['One', 'Two', 'Three', 'Four']))


def resizeAndPad(img, size, padColor=0):
    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw:  # shrinking image
        interp = cv2.INTER_AREA
    else:  # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w / h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1:  # horizontal image
        new_w = sw
        new_h = np.round(new_w / aspect).astype(int)
        pad_vert = (sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1:  # vertical image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = (sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else:  # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor,
                                              (list, tuple, np.ndarray)):  # color image but only one color provided
        padColor = [padColor] * 3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT,
                                    value=padColor)

    return scaled_img


def next_q():
    canvas.itemconfig(headline, text='you can put your next question here!')
    return


def get_x_y():
    x_in_camera = 0
    y_in_camera = 0

    frame = pipeline.wait_for_frames()
    depth_frame = frame.get_depth_frame()
    color_frame = frame.get_color_frame()
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    colorizer = rs.colorizer()
    colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, None, 0.175, 0), cv2.COLORMAP_JET)

    img = colorized_depth
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    hsv_split = np.concatenate((h, s, v), axis=1)

    ret, min_sat = cv2.threshold(s, 40, 255, cv2.THRESH_BINARY)

    ret, max_hue = cv2.threshold(h, 15, 255, cv2.THRESH_BINARY_INV)

    final = cv2.bitwise_and(min_sat, max_hue)

    gray3 = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2GRAY)  # alpha
    hands3 = hand_cascade.detectMultiScale(gray3, 1.1, 5)
    for (x, y, w, h) in hands3:
        cv2.rectangle(depth_colormap, (x, y), (x + w, y + h), (255, 255, 0, 4))  # Blue
        cv2.putText(depth_colormap, 'alpha', org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=(255, 255, 0, 2))

    gray2 = final  # b&W
    hands2 = hand_cascade.detectMultiScale(gray2, 1.1, 5)
    for (x, y, w, h) in hands2:
        cv2.rectangle(final, (x, y), (x + w, y + h), (255, 0, 0, 4))  # Red color
        cv2.putText(final, 'b&W', org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=(255, 0, 0, 4))

    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)  # rgb
    hands = hand_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
    for (x, y, w, h) in hands:
        cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0, 2))  # Green
        cv2.putText(color_image, 'RGB', org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=(0, 255, 0, 2))
        x_in_camera = (x + w / 2)
        y_in_camera = (y + h / 2)
    # Show images
    final_new = cv2.cvtColor(final, cv2.COLOR_GRAY2RGB)
    font = cv2.FONT_HERSHEY_SIMPLEX
    images = np.hstack((color_image, final_new, depth_colormap))
    h, w = images.shape[:2]
    aspect = w / h
    scaled_images = resizeAndPad(images, (240, 1280), 127)
    return x_in_camera, y_in_camera, scaled_images

def ChangeFonts(selected):
    for i in range(6):
        if i == selected: continue
        if iteration_in[i] != 0:
            iteration_in[i] -= 1
            if i<4:
                font_in[i] -= how_mach_make_to_font_bigger
                canvas.itemconfig(answs[i], font=("Purisa", font_in[i]))
            elif i == 4:
                font_in[i] -= how_mach_make_to_font_bigger
                canvas.itemconfig(headline, font=("Purisa", font_in[i]))
            else: # back
                font_in[i] += how_mach_make_to_font_bigger
                canvas.itemconfig(poligon, fill='#{:02X}{:02X}{:02X}'.format(font_in[5], font_in[5], font_in[5]))

    iteration_in[selected] += 1
    if selected < 5:
        font_in[selected] += how_mach_make_to_font_bigger
    else:
        font_in[selected] -= how_mach_make_to_font_bigger
    if selected < 4:
        canvas.itemconfig(answs[selected], font=("Purisa", font_in[selected]))
    elif selected == 4:
        canvas.itemconfig(headline, font=("Purisa", font_in[4]))
    else:  # back
        canvas.itemconfig(poligon, fill='#{:02X}{:02X}{:02X}'.format(font_in[5], font_in[5], font_in[5]))

def PrintAnswers():
    for i,q in enumerate(Questionair):
        print(i+1,") ", q._text)
        for j,a in enumerate(q._answers):
            print("\t", j+1,": ", q._answers[j])
        print("your choice: ", q._choice+1)

def move():
    global x
    global y
    global cur_q
    global iteration_in
    global font_in
    global Rsel
    result = get_x_y()
    new_x = result[0]
    new_y = result[1]
    images = result[2]

    if new_x == 0 and new_y == 0: # מצב שבו אין זיהוי
        canvas.coords(image, -500, -500)
        root.after(30, move) # קרא לפונקציית move בעוד שלושים מילי שניה
        return

    x = 640 - 2 * (new_x - 640) # 1 זה הרגישות
    y = 2 * (new_y - 360) + 360
    canvas.coords(image, x, y)

    q = Questionair[cur_q]
    if y < 120:
        if x < 960:
            type=4
        else:
            type=5
    else:
        if q._type == QT_2ANS:
            if x<640:
                type=0
            else:
                type=1
        elif q._type == QT_4ANS:
            if y<420:
                if x < 640:
                    type = 0
                else:
                    type = 1
            else:
                if x < 640:
                    type = 2
                else:
                    type = 3
        else:   # ruller
            type=0
            newRsel = int((x-27) // 130)
            if (newRsel < 0 or newRsel > 8):
                root.after(30, move)
                return
            if (newRsel!=Rsel):
                Rsel=newRsel
                canvas.itemconfig(answs[0], text=Rsel+1)
                font_in[0]=36
                iteration_in[0]=0

    if iteration_in[type] == 20:
        if type<4:
            canvas.coords(rects[type], 0, 120, 1280, 720)
            canvas.tag_raise(rects[type])
            canvas.coords(answs[type], 640, 300)
            canvas.tag_raise(answs[type])
            canvas.coords(image, -500, -500)
            if q._type==QT_RULLER: q._choice = Rsel
            else: q._choice = type
            cur_q+=1
        elif type==5 and cur_q>0:
            cur_q-=1
        root.after(2000, PrepareQuestion)
        return

    if type!=5 or cur_q>0:
        ChangeFonts(type)

    root.after(30, move)

def PrepareQuestion(loop = True):
    global headline, poligon, rects, answs  # השאלה, רשימה המכילה קורדינטות של כל תשובה בהתאם לסוג השאלה, רשימת התשובות-התשובות האופציונליות שאפשר לבחור מתוכן
    global iteration_in, font_in
    iteration_in = [0, 0, 0, 0, 0, 0]
    font_in = [36, 36, 36, 36, 24, 200]

    if cur_q >= len(Questionair):
        root.quit()
        return;

    q = Questionair[cur_q] # רשימת השאלות שאני מוסיפה
    # headline
    if headline!=None:
        canvas.delete(headline)
    headline = canvas.create_text(480, 60, text=q._text, font=("Purisa", 24))
    canvas.itemconfig(poligon, fill='#{:02X}{:02X}{:02X}'.format(font_in[5], font_in[5], font_in[5]))
    # answers area
    for i in range(len(rects)):
        canvas.delete(rects[i])
        canvas.delete(answs[i])
    rects = []
    answs = []
    if q._type == QT_2ANS:
        rects.append(canvas.create_rectangle(0, 120, 640, 720, fill="Green"))
        rects.append(canvas.create_rectangle(640, 120, 1280, 720, fill="Red"))
        answs.append(canvas.create_text(320, 420, text=q._answers[0], font=("Purisa", 36), justify=CENTER))
        answs.append(canvas.create_text(960, 420, text=q._answers[1], font=("Purisa", 36), justify=CENTER))
    elif q._type == QT_4ANS:
        rects.append(canvas.create_rectangle(0, 120, 640, 420, fill="Green"))
        rects.append(canvas.create_rectangle(640, 120, 1280, 420, fill="Red"))
        rects.append(canvas.create_rectangle(0, 420, 640, 720, fill="Blue"))
        rects.append(canvas.create_rectangle(640, 420, 1280, 720, fill="Yellow"))
        answs.append(canvas.create_text(320, 270, text=q._answers[0], font=("Purisa", 36), justify=CENTER))
        answs.append(canvas.create_text(960, 270, text=q._answers[1], font=("Purisa", 36), justify=CENTER))
        answs.append(canvas.create_text(320, 570, text=q._answers[2], font=("Purisa", 36), justify=CENTER))
        answs.append(canvas.create_text(960, 570, text=q._answers[3], font=("Purisa", 36), justify=CENTER))
    elif q._type == QT_RULLER:
        rects.append(canvas.create_rectangle(0, 120, 1280, 720, fill="white"))
        Rsel=0
        answs.append(canvas.create_text(640, 200, text="1", font=("Purisa", 36), justify=CENTER, fill="black"))
        colors=["#aaaaff","#8888ee","#6666dd","#4444cc","#2222bb","#0000aa","#000099","#000088","#000077"]
        for i in range(9):
            lx = 55 + i * 130
            rects.append(canvas.create_rectangle(lx, 400, lx + 130, 600, fill=colors[i]))
            answs.append(canvas.create_text(lx + 65, 500, text=i+1, font=("Purisa", 48), justify=CENTER, fill="white"))
        # rects.append(canvas.create_line(55, 500, 55, 400, width=10))

        # rects.append(canvas.create_rectangle(960, 120, 1280, 720, fill="Yellow"))
        # answs.append(canvas.create_text(640, 200, text="5", font=("Purisa", 36), justify=CENTER, fill="black"))
        # answs.append(canvas.create_text(480, 420, text=q._answers[1], font=("Purisa", 36), justify=CENTER))
        # answs.append(canvas.create_text(800, 420, text=q._answers[2], font=("Purisa", 36), justify=CENTER))
        # answs.append(canvas.create_text(1120, 420, text=q._answers[3], font=("Purisa", 36), justify=CENTER))

    canvas.tag_raise(image) # פונקציה שמביאה את היד לקדמת הקנבס ככה ששאר האובייקטים על המסך לא יסתירו אותה

    if loop:
        root.after(30, move)


pipeline = rs.pipeline()
config = rs.config()
#config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
#config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.any, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.any, 30)
# Start streaming
pipeline.start(config)

# cap = cv2.VideoCapture(0)
hand_cascade = cv2.CascadeClassifier('palm_v4.xml')

canvas = Canvas(root, width=1280, height=720, bg="white")
canvas.pack()

headline_rectangle = canvas.create_rectangle(0,0,960,120, fill="white")
# headline = canvas.create_text(640, 60, text='Move your hand aside', font=("Purisa", 24))

back_rectangle = canvas.create_rectangle(960,0,1280,120, fill="#e0e0e0")
poligon=canvas.create_polygon(1000,30,1200,60,1000,90, fill='#{:02X}{:02X}{:02X}'.format(font_in[5],font_in[5],font_in[5]))

img = PhotoImage(file="hand icon.png")
image = canvas.create_image(x, y, image=img)
cur_q = 0
PrepareQuestion(False)
time.sleep(5)  # delay for 5 second
root.after(30, move)
root.mainloop()
PrintAnswers()
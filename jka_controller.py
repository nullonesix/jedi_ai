import torch
from torchvision.io import read_image
import win32gui
import time
from PIL import ImageGrab
import numpy as np
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from jka_model import Net
import keyboard
import mouse
import ctypes as cts
import pynput

def set_pos(dx, dy):
    # pos = queryMousePosition()
    # x, y = pos['x'] + dx, pos['y'] + dy
    # x = 1 + int(x * 65536./Wd)
    # y = 1 + int(y * 65536./Hd)
    extra = cts.c_ulong(0)
    ii_ = pynput._util.win32.INPUT_union()
    # ii_.mi = pynput._util.win32.MOUSEINPUT(x, y, 0, (0x0001 | 0x8000), 0, cts.cast(cts.pointer(extra), cts.c_void_p))
    ii_.mi = pynput._util.win32.MOUSEINPUT(dx, dy, 0, (0x0001), 0, cts.cast(cts.pointer(extra), cts.c_void_p))
    command=pynput._util.win32.INPUT(cts.c_ulong(0), ii_)
    cts.windll.user32.SendInput(1, cts.pointer(command), cts.sizeof(command))

weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1

model = Net(in_channels=3)
model.load_state_dict(torch.load('jka_model2.pt'))
model.eval()

while True:

    hwnd = win32gui.FindWindow(None, 'EternalJK')
    win32gui.SetForegroundWindow(hwnd)
    bbox = win32gui.GetWindowRect(hwnd)
    bbox = (bbox[0], bbox[1], bbox[0]+850, bbox[1]+650)
    img = ImageGrab.grab(bbox)
    img.save('current.png')
    x = read_image('current.png')
    x = weights.transforms()(x)
    x = x.unsqueeze(0)
    # img.save('current.png')
    # x = torch.Tensor([np.array(img)])
    # x = torch.permute(x, (0, 3, 1, 2))
    # print(x.shape)
    
    output = model(x)
    print(output)
    w, a, s, d, f, e, r, space, ctrl, mouse_left, mouse_middle, mouse_right, mouse_deltaX, mouse_deltaY = output

    keyboard.release('w, a, s, d, f, e, r, space, ctrl')
    mouse.release(button='left')
    mouse.release(button='middle')
    mouse.release(button='right')

    if keyboard.is_pressed('c'):
        break

    pressed = []

    # if np.argmax(w[0].tolist()):
    #     pressed.append('w')
    # if np.argmax(a[0].tolist()):
    #     pressed.append('a')
    # if np.argmax(s[0].tolist()):
    #     pressed.append('s')
    # if np.argmax(d[0].tolist()):
    #     pressed.append('d')
    # if np.argmax(f[0].tolist()):
    #     pressed.append('f')
    # if np.argmax(e[0].tolist()):
    #     pressed.append('e')
    # if np.argmax(r[0].tolist()):
    #     pressed.append('r')
    # if np.argmax(space[0].tolist()):
    #     pressed.append('space')
    # if np.argmax(ctrl[0].tolist()):
    #     pressed.append('ctrl')

    if np.random.sample() > torch.exp(w)[0].tolist()[0]:
        pressed.append('w')
    if np.random.sample() > torch.exp(a)[0].tolist()[0]:
        pressed.append('a')
    if np.random.sample() > torch.exp(s)[0].tolist()[0]:
        pressed.append('s')
    if np.random.sample() > torch.exp(d)[0].tolist()[0]:
        pressed.append('d')
    if np.random.sample() > torch.exp(f)[0].tolist()[0]:
        pressed.append('f')
    # if np.random.sample() > torch.exp(e)[0].tolist()[0]:
    #     pressed.append('e')
    # if np.random.sample() > torch.exp(r)[0].tolist()[0]:
    #     pressed.append('r')
    if np.random.sample() > torch.exp(space)[0].tolist()[0]:
        pressed.append('space')
    if np.random.sample() > torch.exp(ctrl)[0].tolist()[0]:
        pressed.append('ctrl')

    print(pressed)    

    if pressed:
        keyboard.press(','.join(pressed))

    if np.random.sample() > torch.exp(mouse_left)[0].tolist()[0]:
        mouse.press(button='left')
    if np.random.sample() > torch.exp(mouse_middle)[0].tolist()[0]:
        mouse.press(button='middle')
    if np.random.sample() > torch.exp(mouse_left)[0].tolist()[0]:
        mouse.press(button='right')

    set_pos(round(100.0*mouse_deltaX.tolist()[0][0]), round(100.0*mouse_deltaY.tolist()[0][0]))

    # dx = mouse_deltaX.tolist()[0][0]
    # dy = mouse_deltaY.tolist()[0][0]

    # if dx > 1.0:
    #     dx = round(dx)
    # elif np.random.sample() > mouse_deltaX.tolist()[0][0]:
    #     dx = 0
    # else:


    
    
    
#!/usr/bin/env python

import ctypes as cts
import ctypes.wintypes as wts
import sys
import time

import ctypes_wrappers as cws

import pynput
import mss

import win32api
import win32con
import keyboard
import mouse
import csv
import win32gui
from PIL import ImageGrab


HWND_MESSAGE = -3

WM_QUIT = 0x0012
WM_INPUT = 0x00FF
WM_KEYUP = 0x0101
WM_CHAR = 0x0102

HID_USAGE_PAGE_GENERIC = 0x01

RIDEV_NOLEGACY = 0x00000030
RIDEV_INPUTSINK = 0x00000100
RIDEV_CAPTUREMOUSE = 0x00000200

RID_HEADER = 0x10000005
RID_INPUT = 0x10000003

RIM_TYPEMOUSE = 0
RIM_TYPEKEYBOARD = 1
RIM_TYPEHID = 2

PM_NOREMOVE = 0x0000

xys = []
deltaX = 0
deltaY = 0

def wnd_proc(hwnd, msg, wparam, lparam):
    global deltaX, deltaY
    print(f"Handle message - hwnd: 0x{hwnd:016X} msg: 0x{msg:08X} wp: 0x{wparam:016X} lp: 0x{lparam:016X}")
    if msg == WM_INPUT:
        size = wts.UINT(0)
        res = cws.GetRawInputData(cts.cast(lparam, cws.PRAWINPUT), RID_INPUT, None, cts.byref(size), cts.sizeof(cws.RAWINPUTHEADER))
        if res == wts.UINT(-1) or size == 0:
            print_error(text="GetRawInputData 0")
            return 0
        buf = cts.create_string_buffer(size.value)
        res = cws.GetRawInputData(cts.cast(lparam, cws.PRAWINPUT), RID_INPUT, buf, cts.byref(size), cts.sizeof(cws.RAWINPUTHEADER))
        if res != size.value:
            print_error(text="GetRawInputData 1")
            return 0
        #print("kkt: ", cts.cast(lparam, cws.PRAWINPUT).contents.to_string())
        ri = cts.cast(buf, cws.PRAWINPUT).contents
        #print(ri.to_string())
        head = ri.header
        print(head.to_string())
        #print(ri.data.mouse.to_string())
        #print(ri.data.keyboard.to_string())
        #print(ri.data.hid.to_string())
        if head.dwType == RIM_TYPEMOUSE:
            data = ri.data.mouse
        elif head.dwType == RIM_TYPEKEYBOARD:
            data = ri.data.keyboard
            if data.VKey == 0x1B:
                cws.PostQuitMessage(0)
        elif head.dwType == RIM_TYPEHID:
            data = ri.data.hid
        else:
            print("Wrong raw input type!!!")
            return 0
        print(data.to_string())
        # print('here', data.lLastX)
        if hasattr(data, 'lLastX'):
            # xys.append((data.lLastX, data.lLastY))
            deltaX, deltaY = data.lLastX, data.lLastY
    return cws.DefWindowProc(hwnd, msg, wparam, lparam)


def print_error(code=None, text=None):
    text = text + " - e" if text else "E"
    code = cws.GetLastError() if code is None else code
    print(f"{text}rror code: {code}")


def register_devices(hwnd=None):
    flags = RIDEV_INPUTSINK  # @TODO - cfati: If setting to 0, GetMessage hangs
    generic_usage_ids = (0x01, 0x02, 0x04, 0x05, 0x06, 0x07, 0x08)
    devices = (cws.RawInputDevice * len(generic_usage_ids))(
        *(cws.RawInputDevice(HID_USAGE_PAGE_GENERIC, uid, flags, hwnd) for uid in generic_usage_ids)
    )
    #for d in devices: print(d.usUsagePage, d.usUsage, d.dwFlags, d.hwndTarget)
    if cws.RegisterRawInputDevices(devices, len(generic_usage_ids), cts.sizeof(cws.RawInputDevice)):
        print("Successfully registered input device(s)!")
        return True
    else:
        print_error(text="RegisterRawInputDevices")
        return False


def main(*argv):
    wnd_cls = "SO049572093_RawInputWndClass"
    wcx = cws.WNDCLASSEX()
    wcx.cbSize = cts.sizeof(cws.WNDCLASSEX)
    #wcx.lpfnWndProc = cts.cast(cws.DefWindowProc, cts.c_void_p)
    wcx.lpfnWndProc = cws.WNDPROC(wnd_proc)
    wcx.hInstance = cws.GetModuleHandle(None)
    wcx.lpszClassName = wnd_cls
    #print(dir(wcx))
    res = cws.RegisterClassEx(cts.byref(wcx))
    if not res:
        print_error(text="RegisterClass")
        return 0
    hwnd = cws.CreateWindowEx(0, wnd_cls, None, 0, 0, 0, 0, 0, 0, None, wcx.hInstance, None)
    if not hwnd:
        print_error(text="CreateWindowEx")
        return 0
    #print("hwnd:", hwnd)
    if not register_devices(hwnd):
        return 0
    msg = wts.MSG()
    pmsg = cts.byref(msg)
    print("Start loop (press <ESC> to exit)...")
    pmsgs = []
    t0 = time.time()
    count = 0
    while res := cws.GetMessage(pmsg, None, 0, 0):
        # print('here', cws.PeekMessage(pmsg))
        # if time.time() - t0 > 5:
        #     break

        if res < 0:
            print_error(text="GetMessage")
            break
        cws.TranslateMessage(pmsg)
        cws.DispatchMessage(pmsg)
        # pmsgs.append(pmsg)

        with open('./data/index.txt') as file:
            n = file.read()
            print('n', n)

        hwnd = win32gui.FindWindow(None, 'EternalJK')
        win32gui.SetForegroundWindow(hwnd)
        bbox = win32gui.GetWindowRect(hwnd)
        bbox = (bbox[0], bbox[1], bbox[0]+850, bbox[1]+650)
        img = ImageGrab.grab(bbox)
        img_name = n+'.png'
        img.save('./data/train/'+img_name)

        with open('./data/index.txt', 'w') as file:
            file.write(str(int(n)+1))

        w = 1 if keyboard.is_pressed('w') else 0
        a = 1 if keyboard.is_pressed('a') else 0
        s = 1 if keyboard.is_pressed('s') else 0
        d = 1 if keyboard.is_pressed('d') else 0

        f = 1 if keyboard.is_pressed('f') else 0
        e = 1 if keyboard.is_pressed('e') else 0
        r = 1 if keyboard.is_pressed('r') else 0

        space = 1 if keyboard.is_pressed('space') else 0
        ctrl = 1 if keyboard.is_pressed('ctrl') else 0

        left = 1 if mouse.is_pressed('left') else 0
        middle = 1 if mouse.is_pressed('middle') else 0
        right = 1 if mouse.is_pressed('right') else 0

        with open('./data/data.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([img_name, w, a, s, d, f, e, r, space, ctrl, left, middle, right, deltaX, deltaY])

        print('w', w, 'a', a, 's', s, 'd', d)
        print('f', f, 'e', e, 'r', r)
        print('space', space, 'ctrl', ctrl)
        print('left', left, 'middle', middle, 'right', right)
        print('deltaX', deltaX, 'deltaY', deltaY)

        count += 1
        t1 = time.time()
        if t1 - t0 > 1:
            print('count', count)
            count = 0
            t0 = time.time()

        # break
        

    # return pmsgs

sct = mss.mss()
Wd, Hd = sct.monitors[1]["width"], sct.monitors[1]["height"]
SendInput = cts.windll.user32.SendInput
PUL = cts.POINTER(cts.c_ulong)

from ctypes import windll, Structure, c_long, byref


class POINT(Structure):
    _fields_ = [("x", c_long), ("y", c_long)]



def queryMousePosition():
    pt = POINT()
    windll.user32.GetCursorPos(byref(pt))
    return { "x": pt.x, "y": pt.y}

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

if __name__ == "__main__":
    print("Python {:s} {:03d}bit on {:s}\n".format(" ".join(elem.strip() for elem in sys.version.split("\n")),
                                                    64 if sys.maxsize > 0x100000000 else 32, sys.platform))
    # rc = main(*sys.argv[1:])
    pmsgs = main(*sys.argv[1:])
    # for xy in xys:
    #     time.sleep(0.01)
    #     set_pos(xy[0], xy[1])
    # print(len(xys))
    print("\nDone.\n")
    # sys.exit(rc)
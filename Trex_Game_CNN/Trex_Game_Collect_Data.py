#Data collection from the game
import keyboard
import uuid #We added this library to take screenshots from the game
from PIL import Image
from mss import mss #This library takes the relevant pixels from the screen and frames them
import time
"""
http://www.trex-game.skipser.com/
"""
mon = {"top": 490, "left": 750, "width": 210, "height": 140}
sct = mss()

i = 0

def record_screen(record_id, key):
    global i
    i += 1
    print("{}: {}".format(key, i))
    img = sct.grab(mon)
    im = Image.frombytes("RGB", img.size, img.rgb)
    im.save("./img/{}_{}_{}.png".format(key, record_id, i))

is_exit = False

def exit():
    global is_exit
    is_exit = True

keyboard.add_hotkey("esc", exit)
record_id = uuid.uuid4()

while True:
    if is_exit: break
    try:
        if keyboard.is_pressed(keyboard.KEY_UP):
            record_screen(record_id, "up")
            time.sleep(0.1)
        elif keyboard.is_pressed(keyboard.KEY_DOWN):
            record_screen(record_id, "down")
            time.sleep(0.1)
        elif keyboard.is_pressed("right"):
            record_screen(record_id, "right")
            time.sleep(0.1)
    except RuntimeError: continue

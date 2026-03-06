import time
from pynput.keyboard import Controller, Key
time.sleep(1)
keyboard = Controller()
with keyboard.pressed(Key.ctrl):
    with keyboard.pressed(Key.shift):
        keyboard.press('t')
        keyboard.release('t')

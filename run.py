###run.py is used to execute commands when terminal action is restricted

import subprocess
import sys
import os


env = os.environ.copy()
env["FORCE_COLOR"] = "1"
subprocess.run([sys.executable, "main.py"],env=env)

'''
subprocess.run([sys.executable, "pip", "install", "-r", "requirements.txt"])
'''
"""
env = os.environ.copy()
env["FORCE_COLOR"] = "1"
subprocess.run([sys.executable, "main.py", "--symbol", "BTCUSDT", "--tf", "1H-15m", "--watch", "--interval", "300"],env=env)
"""
'''
###---Bot start---
subprocess.run([sys.executable, "bot.py"])
'''

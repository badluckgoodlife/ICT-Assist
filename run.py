###run.py is used to execute commands when terminal action is restricted

import subprocess
import sys
import os

'''
subprocess.run([sys.executable, "pip", "install", "-r", "requirements.txt"])
'''
'''
---Direct Trigger---
env = os.environ.copy()
env["FORCE_COLOR"] = "1"
subprocess.run([sys.executable, "main.py", "--symbol", "ETHUSDT", "--tf", "1H-15m", "--watch", "--interval", "300"],env=env)
'''
'''
###---Bot start---
subprocess.run([sys.executable, "bot.py"])
'''

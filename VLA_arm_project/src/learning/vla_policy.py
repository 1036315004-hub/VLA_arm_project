# 导入EC63 SDK的核心类（根据你的SDK结构调整）
import sys
import os
# Ensure we load the local 'elite' package from 'eliterobot' folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'eliterobot'))

print("Starting...")
from elite import EC
import time

# 初始化仿真模式（无需连接真实硬件）
try:
    print("Initing EC...")
    # Enable simulation mode
    arm = EC(sim=True)
    print("✅ EC63 SDK 本地导入成功！虚拟环境配置完成")
    time.sleep(10) # Keep window open
except Exception as e:
    print(f"⚠️ 初始化提示：{e}")
    import traceback
    traceback.print_exc()


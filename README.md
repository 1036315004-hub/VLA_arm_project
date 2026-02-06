[English](#english-version) | [中文说明](#中文说明)
------

## 中文说明

# 项目：面向桌面物体的视觉-语言-动作 (VLA) 机械臂闭环控制系统

## 第一部分：项目综述与问题定义 (Introduction & Problem Definition)

- **1.1 研究背景：** 机器人从专才向通用大模型 (Generalist Robot) 的演进需求
- **1.2 核心目标：** 构建一个能够理解自然语言指令并根据多模态视觉反馈执行精准抓取任务的端到端策略
- **1.3 系统挑战：** 异质数据对齐（视觉/语言）、高质量训练数据的自动获取、动作空间的连续性表达


## 第二部分：自动化数据采集与 Oracle 监督 (Automated Data Collection & Oracle Supervision)

- **2.1 基于 Oracle 的自动化标注系统：** 利用特权信息（Oracle）实现端到端的专家演示自动生成
- **2.2 三阶段动作执行逻辑 (Keyframe Strategy)：** 悬停 (Hover) ; 接触预备 (Pre-contact) ; 物理接触 (Contact) 的三阶段关键帧定义
- **2.3 动态场景随机化：** 包含物体位姿、光照参数、障碍物分布的自动重置机制（Domain Randomization 雏形）


## 第三部分：科研级数据质量门控系统 (Quality Gate System)

- **3.1 五维度质量评估模型：**
  - **Gate A:** 关键帧完整性校验
  - **Gate B:** 运动收敛精度 (Convergence Distance) 评估
  - **Gate C:** 物理接触检测 (Contact Detection) 验证
  - **Gate D:** 深度特征有效性与图像质量检查
  - **Gate E:** 最小有效步数约束
- **3.2 离线数据清洗流：** 确保进入训练集的每一条轨迹均具备高成功率和物理一致性

## 第四部分：多模态数据预处理流水线 (Data Preprocessing Pipeline)

- **4.1 语言特征空间映射：** 利用 CLIP Text Encoder 实现自然语言指令的高维语义嵌入
- **4.2 视觉数据标准化：** 针对 DINOv2 的 ImageNet 归一化与 224x224 分辨率重采样
- **4.3 动作空间归一化处理：** 将 7D 动作（XYZ + 四元数）映射至 [-1, 1] 空间，解决位置与姿态的数值量级失衡

## 第五部分：VLA 策略网络架构设计 (Policy Network Architecture)

- **5.1 混合主干网络 (Hybrid Backbone)：**
  - **视觉特征提取：** 冻结的 DINOv2 (ViT-S/14) 全局语义提取
  - **深度感知模块：** 轻量级深度图编码器（CNN）捕获空间几何信息
- **5.2 多模态特征融合层 (RGB-D-T Fusion)：** RGB、深度、文本特征的高维拼接与非线性映射
- **5.3 解耦动作预测头 (Decoupled Action Heads)：**
  - 基于 Tanh 激活的位置预测分支
  - 基于 L2 归一化的旋转（四元数）预测分支

## 第六部分：当下执行项（Current execution item）

- **6.1 采集性能分析：** 自动化采集的成功率、采纳率与失败因子分析
- **6.2 模型评估指标：** 不同指令下的任务执行成功率、轨迹偏差 (Trajectory Error) 统计
- **6.3 可视化分析：** 关键帧检查点与机器人执行过程的同步演示
- **6.4 后续改进：** 引入视觉变换器 (Vision Transformer) 的 Fine-tuning、探索端到端的交互式学习策略

 ---

## English Version
# Project: Visual-Language-Action (VLA) Closed-Loop Control System for Desktop Objects for Mechanical Arms 
## Part One: Project Overview and Problem Definition 
- **1.1 Research Background:** The evolution requirement of robots from specialized models to generalist models (Generalist Robot)
- **1.2 Core Objective:** To build an end-to-end strategy that can understand natural language instructions and perform precise grasping tasks based on multi-modal visual feedback
- **1.3 System Challenges:** Heterogeneous data alignment (visual/language), automatic acquisition of high-quality training data, continuous expression of action space 

## Part Two: Automated Data Collection and Oracle Supervision 
- **2.1 Oracle-based Automated Annotation System:** Utilizing privileged information (Oracle) to automatically generate end-to-end expert demonstrations
- **2.2 Three-stage Action Execution Logic (Keyframe Strategy):** Three-stage keyframes defined as Hover; Pre-contact; Contact
- **2.3 Dynamic Scene Randomization:** Automatic reset mechanism for object poses, lighting parameters, and obstacle distribution (Prototype of Domain Randomization) 

## Part Three: Research-level Data Quality Gate Control System 
- **3.1 Five-Dimensional Quality Assessment Model:**
  - **Gate A:** Verification of key frame integrity
  - **Gate B:** Evaluation of motion convergence accuracy (Convergence Distance)
  - **Gate C:** Validation of physical contact detection (Contact Detection)
  - **Gate D:** Examination of the effectiveness of depth features and image quality
  - **Gate E:** Constraint of minimum effective step count
  - **3.2 Offline Data Cleaning Flow:** Ensure that every trajectory entering the training set has a high success rate and physical consistency 
## Part Four: Multi-modal Data Preprocessing Pipeline 
- **4.1 Language Feature Space Mapping:** Utilizing the CLIP Text Encoder to achieve high-dimensional semantic embeddings for natural language instructions
- **4.2 Visual Data Normalization:** Applying ImageNet normalization for DINOv2 and resampling to a resolution of 224x224
- **4.3 Action Space Normalization Processing:** Mapping 7D actions (XYZ + quaternion) to the [-1, 1] space to address the imbalance in numerical magnitudes of position and posture 
## Part Five: VLA Policy Network Architecture Design 
- **5.1 Hybrid Backbone:**
- **Visual Feature Extraction:** Frozen DINOv2 (ViT-S/14) for global semantic extraction
- **Depth Perception Module:** Lightweight depth map encoder (CNN) for capturing spatial geometric information
- **5.2 RGB-D-T Fusion Layer:** High-dimensional concatenation and nonlinear mapping of RGB, depth, and text features
- **5.3 Decoupled Action Prediction Heads:**
- Position prediction branch based on Tanh activation
- Rotation (quaternion) prediction branch based on L2 normalization 
## Part VI: Current Execution Item 
- **6.1 Performance Analysis of Collection:** Analysis of the success rate, adoption rate, and failure factors of automated collection
- **6.2 Evaluation Metrics of Models:** Statistics on the success rate of task execution and trajectory deviation under different instructions
- **6.3 Visual Analysis:** Synchronous demonstration of key frame checkpoints and the robot's execution process
- **6.4 Future Improvements:** Fine-tuning with the Vision Transformer, exploration of end-to-end interactive learning strategies 
---

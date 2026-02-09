<img width="2784" height="1536" alt="current system" src="https://github.com/user-attachments/assets/4195dc54-c4c3-4675-a407-e76b810df60b" />

[English](#english-version) | [中文说明](#中文说明)  
---
## English-version
#  Six-axis Mechanical Arm Assistant for Generalized Desktop Object Cleaning
Core Objective: Develop a highly generalized desktop object cleaning mechanical arm assistant that can perform operations such as grasping, positioning, and organizing desktop objects based on natural language instructions, and complete technological iterations and scenario implementation.
## Part One: Project Overview and Problem Definition 
- **1.1 Research Background:** The evolution requirement of robots from specialized models to generalist models (Generalist Robot); The gap in generalization operation technology for robotic arms for cleaning desktop objects
- **1.2 Core Objectives:** To build an end-to-end strategy that can understand natural language instructions and perform precise grasping tasks based on multi-modal visual feedback
- **1.3 System Challenges:** Heterogeneous data alignment (visual/language), automatic acquisition of high-quality training data, continuous expression of action space; Generalization of complex desktop cleaning scenarios (multiple objects / occlusion / thin objects) 

## Part Two: Automated Data Collection and Oracle Oversight 
- **2.1 Oracle-based Automated Annotation System:** Utilizing privileged information (Oracle) to achieve end-to-end automatic generation of expert demonstrations, and adapting to the 6D pose truth acquisition of desktop objects.
- **2.2 Three-stage Action Execution Logic (Keyframe Strategy):** Three-stage keyframes defined for hovering (Hover), pre-contact (Pre-contact), and contact (Contact), which align with the grasping and pushing base action logic for desktop cleaning.
- **2.3 Dynamic Scene Randomization:** An automatic reset mechanism for object poses, lighting parameters, and obstacle distribution (an embryonic form of Domain Randomization). 

## Part Three: Data Quality Gatekeeper System 
- **3.1 Five-Dimensional Quality Assessment Model:**
- **Gate A:** Verification of key frame integrity
- **Gate B:** Evaluation of motion convergence accuracy (Convergence Distance), high precision constraints for small-scale operations on the desktop
- **Gate C:** Verification of physical contact (Contact Detection), validation of the effectiveness of soft contact / pushing and pulling of desktop objects
- **Gate D:** Check of depth feature validity and image quality
- **Gate E:** Constraint of minimum effective steps
- **3.2 Offline Data Cleaning Flow:** Ensure that every trajectory entering the training set has high success rate and physical consistency; guarantee the physical feasibility of desktop cleaning operations 
## Part Four: Multimodal Data Preprocessing Pipeline 
- **4.1 Language Feature Space Mapping:** Utilize the CLIP Text Encoder to achieve high-dimensional semantic embeddings for natural language instructions; adapt natural language instructions for desktop cleaning (such as "pick up this blanket" / "grab the scattered pens").
- **4.2 Visual Data Normalization:** Apply ImageNet normalization for DINOv2 and perform 224x224 resolution resampling.
- **4.3 Action Space Normalization Processing:** Map 7D actions (XYZ + quaternion) to the [-1, 1] space to address the numerical magnitude imbalance between position and posture. 
## Part Five: Design of VLA Strategy Network Architecture 
- **5.1 Hybrid Backbone:**
- **Visual Feature Extraction:** Frozen DINOv2 (ViT-S/14) for global semantic extraction
- **Depth Perception Module:** Lightweight depth map encoder (CNN) for capturing spatial geometric information
- **5.2 RGB-D-T Fusion Layer:**
- High-dimensional concatenation and nonlinear mapping of RGB, depth, and text features
- **5.3 Decoupled Action Prediction Heads:**
- Position prediction branch based on Tanh activation
- Rotation (quaternion) prediction branch based on L2 normalization 
## Part VI: Current Execution Items 
- **6.1 Performance Analysis of Collection:** Analysis of the success rate, adoption rate, and failure factors of automated collection
- **6.2 Evaluation Indicators of the Model:** Statistics of task execution success rate and trajectory deviation (Trajectory Error) under different instructions
- **6.3 Visual Analysis:** Synchronous demonstration of key frame checkpoints and robot execution process
- **6.5 Data Optimization:** Collection of real machine desktop cleaning scene data, covering 6D poses of common desktop objects (cups, pens, cards, tissue boxes), joint angles of the robotic arm, and RGB-D images; Fusion of real machine data with existing simulation data, screening of effective samples through a quality gate control system (Gate A-E), to enhance the model's adaptability to the real desktop environment.
- **6.6 Construction of Complex Desktop Scenarios:** In the simulation environment, build complex desktop scenarios that closely resemble reality, including multi-object stacking (such as cups stacked, pens scattered and stacked), object occlusion (such as small objects partially obscured by books), thin objects adhered to the table (such as cards, notes), different desktop materials (wooden, glass, plastic), and random obstacles (such as desktop clutter); Scene parameters support random configuration (object quantity, position, posture, lighting) to simulate the complex constraints of real desktop cleaning.
- **6.7 Fine-tuning of the Model's Visual Module:** Based on real machine desktop image data, conduct lightweight fine-tuning of the existing DINOv2 (ViT-S/14) visual main network to optimize the accuracy of object feature extraction in desktop scenarios; Combine the instruction matching ability of the CLIP text encoder, fine-tune the multimodal fusion layer to improve the mapping accuracy of "natural language instructions - desktop object visual features".
- **6.7 Exploration of Gripper Structure Optimization:** For the cleaning requirements of thin desktop objects and irregular objects (such as curled tissues, irregular ornaments), explore flexible structure grippers (such as silicone-covered grippers), and 2-3 degree-of-freedom simple dexterous hands for selection and adaptation testing; Verify the impact of different gripper structures on the success rate of desktop object grasping, with the current core of execution being "selection testing + simulation verification", without involving large-scale modification of the real machine robotic arm.  
- **6.9 Adaptation Optimization for Sim-to-Real:** Based on real machine desktop operation feedback (such as object grasping slipping, trajectory deviation of pushing and pulling), adjust simulation parameters, including object friction coefficient, desktop support force, mechanical arm joint dynamics parameters, lighting intensity, etc.; Establish a mapping relationship between "real machine errors - simulation parameters", through parameter calibration and error compensation, to reduce the differences between the simulation environment and the real desktop, and improve the stability of model deployment on the real machine.  
<img width="1376" height="768" alt="future research" src="https://github.com/user-attachments/assets/25ee86ee-2243-4df4-8d13-8fa588f08302" />



## 中文说明

# 面向泛化性桌面物体清理的六轴机械臂助手
核心目标：开发泛化性强的桌面物体清理机械臂助手，实现自然语言指令驱动的桌面物体抓取 / 归位 / 整理等操作，完成技术迭代与场景落地。
## 第一部分：项目综述与问题定义 

- **1.1 研究背景：** 机器人从专才向通用大模型 (Generalist Robot) 的演进需求；面向桌面物体清理的机械臂泛化操作技术缺口
- **1.2 核心目标：** 构建一个能够理解自然语言指令并根据多模态视觉反馈执行精准抓取任务的端到端策略
- **1.3 系统挑战：** 异质数据对齐（视觉/语言）、高质量训练数据的自动获取、动作空间的连续性表达；桌面清理的复杂场景泛化（多物体 / 遮挡 / 薄物体）


## 第二部分：自动化数据采集与 Oracle 监督 

- **2.1 基于 Oracle 的自动化标注系统：** 利用特权信息（Oracle）实现端到端的专家演示自动生成，适配桌面物体的 6D 位姿真值采集
- **2.2 三阶段动作执行逻辑 (Keyframe Strategy)：** 悬停 (Hover) ; 接触预备 (Pre-contact) ; 物理接触 (Contact) 的三阶段关键帧定义，贴合桌面清理的抓取 / 推拽基础动作逻辑。
- **2.3 动态场景随机化：** 包含物体位姿、光照参数、障碍物分布的自动重置机制（Domain Randomization 雏形）


## 第三部分：数据质量门控系统 

- **3.1 五维度质量评估模型：**
  - **Gate A:** 关键帧完整性校验
  - **Gate B:** 运动收敛精度 (Convergence Distance) 评估，桌面小范围操作的高精度约束
  - **Gate C:** 物理接触检测 (Contact Detection) 验证，桌面物体软接触 / 推拽的接触有效性
  - **Gate D:** 深度特征有效性与图像质量检查
  - **Gate E:** 最小有效步数约束
- **3.2 离线数据清洗流：** 确保进入训练集的每一条轨迹均具备高成功率和物理一致性；保障桌面清理操作的物理可行性

## 第四部分：多模态数据预处理流水线 

- **4.1 语言特征空间映射：** 利用 CLIP Text Encoder 实现自然语言指令的高维语义嵌入；适配桌面清理的自然语言指令（如 “拿起这个被子”/“抓取散落的笔”）。
- **4.2 视觉数据标准化：** 针对 DINOv2 的 ImageNet 归一化与 224x224 分辨率重采样
- **4.3 动作空间归一化处理：** 将 7D 动作（XYZ + 四元数）映射至 [-1, 1] 空间，解决位置与姿态的数值量级失衡

## 第五部分：VLA 策略网络架构设计

- **5.1 混合主干网络 (Hybrid Backbone)：**
  - **视觉特征提取：** 冻结的 DINOv2 (ViT-S/14) 全局语义提取
  - **深度感知模块：** 轻量级深度图编码器（CNN）捕获空间几何信息
- **5.2 多模态特征融合层 (RGB-D-T Fusion)：** RGB、深度、文本特征的高维拼接与非线性映射
- **5.3 解耦动作预测头 (Decoupled Action Heads)：**
  - 基于 Tanh 激活的位置预测分支
  - 基于 L2 归一化的旋转（四元数）预测分支

## 第六部分：当下执行项

- **6.1 采集性能分析：** 自动化采集的成功率、采纳率与失败因子分析
- **6.2 模型评估指标：** 不同指令下的任务执行成功率、轨迹偏差 (Trajectory Error) 统计
- **6.3 可视化分析：** 关键帧检查点与机器人执行过程的同步演示
- **6.5 数据优化** ：采集实机桌面清理场景数据，涵盖常见桌面物体（杯子、笔、卡片、纸巾盒）的 6D 位姿、机械臂关节角度、RGB-D 图像；将实机数据与现有仿真数据融合，通过质量门控系统（Gate A-E）筛选有效样本，提升模型对真实桌面环境的适配性。
- **6.6 复杂桌面场景构建**：在仿真环境中搭建贴合实际的复杂桌面场景，包含多物体堆叠（如杯子叠放、笔散落堆叠）、物体遮挡（如小物件被书本部分遮挡）、薄物体贴桌（如卡片、便签）、不同桌面材质（木质 / 玻璃 / 塑料）及随机障碍物（如桌面杂物）；场景参数支持随机化配置（物体数量、位置、姿态、光照）模拟真实桌面清理的复杂约束。
- **6.7 模型视觉模块微调**：基于实机桌面图像数据，对现有 DINOv2 (ViT-S/14) 视觉主干网络进行轻量微调，优化桌面场景下物体特征提取精度；结合 CLIP 文本编码器的指令匹配能力，微调多模态融合层，提升 “自然语言指令 - 桌面物体视觉特征” 的映射准确性。
- **6.7 夹爪结构优化探索**：针对桌面薄物体、不规则物体（如卷曲纸巾、异形摆件）的清理需求，探索柔性结构夹爪（如硅胶包裹夹爪）、2-3 自由度简易灵巧手的选型与适配测试；验证不同夹爪结构对桌面物体抓取成功率的影响，当下执行核心为 “选型测试 + 仿真验证”，不涉及实机机械臂大规模改装。  
- **6.9 sim2real适配优化**：基于实机桌面操作反馈（如物体抓取滑落、推拽轨迹偏差），调整 仿真参数，包括物体摩擦系数、桌面支撑力、机械臂关节动力学参数、光照强度等；建立 “实机误差 - 仿真参数” 的映射关系，通过参数校准与误差补偿，缩小仿真环境与真实桌面的差异，提升模型实机部署的稳定性。

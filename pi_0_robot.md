# Policy
## ALOHA policy
### 数据结构定义： make_aloha_example()
用于测试数据策略通道是否正常
### 输入预处理：AlohaInputs.__call__(0)
机器人自身数据(data) -> pi0模型接口
#### decode_aloha(data)
  - decode_state(state)
        state(the training data) = origin_state * joint_flip_mask;
        gripper_to_angualr: 
        ALOHA归一化线性值(0~1) 
            反归一化成物理线性位移(米) 
            用连杆机构的几何关系(余弦定理)算出舵机转角(弧度) 
            按pi0训练时的角度范围重新归一化(0~1)
  - convert_image(img):

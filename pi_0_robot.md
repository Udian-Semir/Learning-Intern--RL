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
  - convert_image(img)

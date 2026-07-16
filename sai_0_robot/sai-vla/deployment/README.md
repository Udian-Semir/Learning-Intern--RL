# Sai0 VLA 部署.

这个目录包含用于部署 Sai0 VLA 模型的工具和服务器。

## 目录结构

```
deployment/
├── Sai0_1_server/        # HTTP 推理服务器
│   ├── server.py         # FastAPI 服务器应用
│   ├── client.py         # Python 客户端工具
│   ├── client_test.py    # Python 客户端测试工具
│   ├── config.yaml       # 配置文件
│   ├── requirements.txt  # 依赖
│   └── README.md         # 详细文档
├── tele/                 # 遥操作服务器
├── tele_parad/           # 遥操作范式服务器
└── README.md             # 本文件
```

## 快速开始

### 1. Sai0 推理服务器

HTTP API 服务器，支持实时动作预测：

```bash
cd Sai0_1_server

# 启动服务器
python server.py \
    --action_head_ckpt /path/to/checkpoint \
    --vlm_backend qwen3-vl \
    --device cuda:0
```

详细文档: [Sai0_1_server/README.md](./Sai0_1_server/README.md)

## 支持的部署方式

### 1. HTTP API 服务器

- ✅ RESTful API
- ✅ 支持单个和批量推理
- ✅ Python 客户端
- ✅ 低延迟

### 2. gRPC 服务器 (TODO)

- 更高性能
- 双向流式传输
- 适合高频率推理

### 3. ROS 节点 (TODO)

- 直接集成到 ROS 生态
- 支持 ROS 消息类型
- 适合机器人控制

## 性能基准

在 NVIDIA A100 GPU 上的推理性能（Qwen3-VL-2B + Flow Matching）:

| 配置 | VLM 时间 | Action Head 时间 | 总时间 | FPS |
|------|----------|------------------|--------|-----|
| 单图像 | ~120ms | ~10ms | ~130ms | ~7.7 |
| 2 图像 | ~150ms | ~10ms | ~160ms | ~6.3 |
| 4 图像 | ~200ms | ~10ms | ~210ms | ~4.8 |

## 最佳实践

1. **GPU 加速**: 始终在 GPU 上运行推理
2. **批量处理**: 对于多客户端场景，使用批量预测
3. **图像预处理**: 在客户端进行图像压缩和调整大小
4. **连接池**: 使用 HTTP 连接池减少开销
5. **监控**: 使用 Prometheus 监控推理延迟和吞吐量

## 故障排除

常见问题和解决方案请参考各服务器的 README.md。

## 贡献

欢迎贡献新的部署方式和优化！

## 许可证

与主项目相同。

# STABLE BASELINE

## 1. 版本记录

- 稳定仓库：`WUHAO19831214/audio-visual-soundfield-tracker-stable`
- 稳定 commit：`6c5c6b7ca23155db58b63794993322fbdeb8f868`
- 来源开发仓库：`WUHAO19831214/audio-visual-soundfield-tracker`
- 对应开发 commit：`6c5c6b7ca23155db58b63794993322fbdeb8f868`
- 来源提交说明：`Stabilize audio runtime and marker tracking`
- 最后验证日期：2026-07-13

该 SHA 同时存在于两个仓库，因此基线关系可直接由 Git commit 校验，不是仅凭日期或 README 描述推断。

## 2. 验证环境

| 项目 | 值 |
| --- | --- |
| 计算机 | MacBook Air, Apple M2, 16 GB |
| 操作系统 | macOS 26.3.1, Build 25D2128 |
| Python | 3.12.13（复用开发仓库 `.venv` 运行验证） |
| 真实设备组合（此前实验） | FaceTime 高清相机、iPhone Continuity Camera、Wireless Mic Rx/USB 音频接收器 |

真实设备名称会随 macOS 枚举变化。表中设备说明表示该项目采用过的实验组合，不代表 2026-07-13 的自动测试重新完成了所有硬件采集。硬件是否可用仍须执行页面预览、麦克风电平和正式采集检查。

## 3. 本次验证结果

```bash
python -m compileall app.py src tests
pytest -q
```

结果：

- Python 编译检查通过；
- `64 passed in 3.96s`；
- Streamlit 使用测试端口 8511 启动，`curl -I` 返回 `HTTP/1.1 200 OK`；
- 未修改任何程序文件。

建议启动验证：

```bash
streamlit run app.py
curl -I http://localhost:8501
```

页面 HTTP 200、摄像头成功读取、麦克风有输入电平、停止后 CSV 非空是四项不同检查，不能互相替代。

## 4. 测试通过的功能范围

自动测试覆盖的主要边界包括：

- 音频归一化、RMS/dBFS、频谱特征；
- 音频流生命周期、队列和错误状态；
- 麦克风与摄像头设备辅助逻辑；
- 视觉—音频最近邻融合和 0.15 秒容差；
- person 追踪与主轨迹选择；
- 指定物体和 HSV 网球标记追踪；
- 轨迹、声学数值/颜色图和声场图生成；
- 应用关键入口和导出行为。

自动测试不证明：摄像头硬件同步、声压级计量精度、任意 macOS 版本的设备枚举、iPhone Continuity Camera 始终可用，或图像像素能直接转换为米。

## 5. 稳定样例数据

本仓库不提交真实采集 CSV、音视频、PNG/JPEG、模型权重或 `.venv`。稳定样例由测试中的合成数组和临时 CSV 生成，保证不含个人图像与设备隐私。

正式实验样例应在独立资料目录归档，并至少记录：

- 文件名与 SHA-256；
- 本基线 commit；
- 设备、系统、采样率和摄像头分辨率；
- 采集动作、时长、融合匹配率；
- `time_diff_sec` 分布；
- 哪些点是实测、哪些图面区域是插值。

## 6. 与开发版差异

开发仓库提交 `71d472c` 和 `3cd8b22` 在本基线之后，增加了双摄网球采集、相机预热/黑帧处理、虚拟三维轨迹、人物相对显示坐标、Plotly 视角、声学三维叠加、棋盘格双目标定和米制三角测量。

本稳定版故意不包含这些实验功能。它的目标是保存 `6c5c6b7` 时已经验证的单摄视觉—音频同步链路，便于出现开发回归时快速比较。

## 7. 升级基线流程

1. 在开发仓库选定明确 commit 或签名 tag；
2. 在目标设备完成 compileall、pytest、Streamlit 页面、摄像头、麦克风和导出检查；
3. 保存验证记录和非隐私样例数据校验值；
4. 对比当前稳定版，列出新增、删除、行为变化和已知问题；
5. 更新本文件与 README 后再发布稳定仓库；
6. 不直接复制开发仓库 README。

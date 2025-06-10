#gazebo中使用

1. 启动gazebo仿真 roslaunch ares_gazebo ares_playground_gazebo.launch
2. 启动跟随程序   roslaunch stage_first OnYourMarkGetSetGo.launch

# Reinforcement Learning Training

该仓库包含一个用于 Velodyne 激光雷达的强化学习示例。运行前需要安装依赖：

```bash
pip install -r requirements.txt
```

然后执行训练脚本：

```bash
python -m rl.train_velodyne_td3_pro
```

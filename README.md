# krBodyAI
krkr集成ncnn引擎调用movenet模型进行人体识别插件

## 编译方式：

1. 下载vulkan-sdk：https://vulkan.lunarg.com/sdk/home
2. 在环境变量中配置VULKAN_HOME指向vulkan-sdk的安装目录。
3. 下载opencv,ncnn预构建包（配置过程比较麻烦，不推荐自行构建）：https://pan.baidu.com/s/1--TzNLMi5R6mS6yWBXPcHg  提取码：sora
4. 打开krBodyAI.sln，生成解决方案，编译完成。

## 运行方式：

1. 把预构建包中的bin文件下所有的文件拷贝到和krBodyAI.dll相同目录下。
2. example文件下有对用的tjs层调用方式（ps：把里面的`Plugins.link("json.dll");`注释掉）

运行krkr.exe主程序，在onframe中即可得到人体17个点的数据每帧回调，x为横坐标，y为纵坐标，score为该点坐标的可信度，数组一共由17个字典组成，样例里面附带了一个识别人体深蹲动作的简单算法（新增跳绳运动算法）

![74dc1761gy1gya2agahz1g20dc07hb2a](https://user-images.githubusercontent.com/22819281/158337264-275d2c27-81f3-4bc8-b393-c987a0cdb6a4.gif)

![录制_2022_03_15_16_22_43_47](https://user-images.githubusercontent.com/22819281/158336670-9a684a75-49ed-477c-9ef2-eb942426b933.gif)

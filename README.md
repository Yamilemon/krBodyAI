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

```javascript
System.inform("I am the shadow of Sora");
var win = new Window();
win.width = 800;
win.height = 600;
win.visible = true;

var lay = new Layer(win, null);
lay.setSize(500,500);
lay.visible = true;
win.add(lay);
lay.font.height = 50;

Plugins.link("krBodyAI.dll");
// Plugins.link("json.dll");
// 第二个参数是姿势分类器的tensorflowlite模型，目前已废弃
bodyAI.init("lightning", "classifier_yoga.tflite");
// 开启摄像头，第一个参数为真时会打开摄像头窗口，第二个参数为真时会画出摄像头中的人体轮廊
bodyAI.openCamera(true, true);
Scripts.execStorage('pose_process.tjs');
var c_system = new Cartesian(320, 120);// 640*480的图像中间点偏上处建立笛卡尔坐标系
var shendun = new rise_down(c_system); // 构建深蹲运动轨迹类
// 摄像头帧回调函数
bodyAI.onFrame = function(bodyData){
    shendun.onReceive(bodyData, shendun);
};
   ```

![74dc1761gy1gya2agahz1g20dc07hb2a](https://user-images.githubusercontent.com/22819281/158337264-275d2c27-81f3-4bc8-b393-c987a0cdb6a4.gif)

![录制_2022_03_15_16_22_43_47](https://user-images.githubusercontent.com/22819281/158336670-9a684a75-49ed-477c-9ef2-eb942426b933.gif)

![录制_2022_03_17_18_30_20_280](https://user-images.githubusercontent.com/22819281/158791549-a1ce15e9-2ea6-4e03-8b0c-c8d12723603d.gif)


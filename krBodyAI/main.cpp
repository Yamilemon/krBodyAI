/*krkr*/
#include "ncbind.hpp"

/*AI*/
#include "net.h"
#include <algorithm>
#include <opencv2/core/core.hpp> 
#include<opencv2/highgui/highgui.hpp> 
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include "cpu.h"
#include "nanodet.h"
//#include "tensorflow/lite/c/c_api.h"
//#pragma comment( lib, "tensorflowlite_c.dll.if.lib" )

using namespace cv;
using namespace std;

/*global item*/
//TfLiteTensor* input_tf;
//TfLiteTensor* output_tf;
//TfLiteInterpreter* interpreter;
static NanoDet* g_nanodet = 0;
static ncnn::Mutex lock;
bool isDrawLine = false;
iTJSDispatch2 *frameCallBack;
int bodyPointNum = 17;
int bodyClassNum = 5;
iTJSDispatch2 *globalBean;

/*static global function*/

static char * wchar2char(const wchar_t* wchar)
{
	char * m_char;
	int len = WideCharToMultiByte(CP_ACP, 0, wchar, wcslen(wchar), NULL, 0, NULL, NULL);
	m_char = new char[len + 1];
	WideCharToMultiByte(CP_ACP, 0, wchar, wcslen(wchar), m_char, len, NULL, NULL);
	m_char[len] = '\0';
	return m_char;
}

static wchar_t* trstring2wchar(const  char *str)
{
	int mystringsize = (int)(strlen(str) + 1);
	WCHAR* wchart = new wchar_t[mystringsize];
	MultiByteToWideChar(CP_ACP, 0, str, -1, wchart, mystringsize);
	return wchart;
}
// 下面是为tensorflowlite预留的代码，我努力过了，但是不能怪了，这玩意在x86上真的很吃屎，不想再努力了
// TODO 有生之年内可以找到32位x86能用的tensorflowlite_c库，要集成进来，脑瘫玩意
/*TfLiteTensor * getOutputTensorByName(TfLiteInterpreter * interpreter, const char * name)
{
	int count = TfLiteInterpreterGetOutputTensorCount(interpreter);
	for (int i = 0; i < count; ++i) {
	TfLiteTensor* ts = (TfLiteTensor*)TfLiteInterpreterGetOutputTensor(interpreter, i);
	//if (!strcmp(ts->name, name)) {
		//return ts;
	//}
	}
	return nullptr;
}
TfLiteTensor * getInputTensorByName(TfLiteInterpreter * interpreter, const char * name)
{
	int count = TfLiteInterpreterGetInputTensorCount(interpreter);
	for (int i = 0; i < count; ++i) {
	TfLiteTensor* ts = TfLiteInterpreterGetInputTensor(interpreter, i);
	//if (!strcmp(ts->name, name)) {
		//return ts;
	//}
	}
	return nullptr;
}
// 加载模型
void initModel(string path) {
	TfLiteModel* model = TfLiteModelCreateFromFile(path.c_str());
	TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
	// 四线程推理
	TfLiteInterpreterOptionsSetNumThreads(options, 4);
	interpreter = TfLiteInterpreterCreate(model, options);
	if (interpreter == nullptr) {
		printf("Failed to create interpreter");
		cout << (path) << endl;
		return;
	}
	// Allocate tensor buffers.
	if (TfLiteInterpreterAllocateTensors(interpreter) != kTfLiteOk) {
		printf("Failed to allocate tensors!");
		return;
	}

	// 输入输出张量
	output_tf = (TfLiteTensor*)TfLiteInterpreterGetOutputTensor(interpreter, 0);
	input_tf = (TfLiteTensor*)TfLiteInterpreterGetInputTensor(interpreter, 0);

	//input_tf = getInputTensorByName(interpreter, "input");
	//output_tf = getOutputTensorByName(interpreter, "MobilenetV3/Predictions/Softmax");

}
// 向前推理
static void forward(std::vector<keypoint> person, int len) {
	float inputVector[51];// 17个点，每个点的x和y还有score分数，加起来是17*3的数组长度
	for (int i = 0; i < person.size(); i++) {
		inputVector[i * 3] = person[i].y;
		inputVector[i * 3 + 1] = person[i].x;
		inputVector[i * 3 + 2] = person[i].score;
	}
	TfLiteTensorCopyFromBuffer(input_tf, inputVector, len * sizeof(float));
	TfLiteInterpreterInvoke(interpreter);
	float logits[5];
	TfLiteTensorCopyToBuffer(output_tf, logits, 5 * sizeof(float));
	float maxV = -1;
	int maxIdx = -1;
	for (int i = 0; i < 5; i++) {
		//if (logits[i] > maxV) {
			//maxV = logits[i];
			//maxIdx = i;
		//}
		//printf("%d->%f\n", i, logits[i]);
	}
	printf("当前动作是tree”的概率为：->%f\n", logits[1]);
	//cout << "类别：" << maxIdx << "，概率：" << maxV << endl;
	/*TfLiteTensorCopyFromBuffer(input_tf, data, len * sizeof(float));
	TfLiteInterpreterInvoke(interpreter);
	float logits[1001];
	TfLiteTensorCopyToBuffer(output_tf, logits, 1001 * sizeof(float));
	float maxV = -1;
	int maxIdx = -1;
	for (int i = 0; i < 1001; ++i) {
		if (logits[i] > maxV) {
			maxV = logits[i];
			maxIdx = i;
		}
		//printf("%d->%f\n", i, logits[i]);
	}
	//cout << "类别：" << maxIdx << "，概率：" << maxV << endl;
}

static int detect_squeezenet(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
	ncnn::Net squeezenet;
	squeezenet.load_param("squeezenet_v1.1.param");
	squeezenet.load_model("squeezenet_v1.1.bin");

	ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 227, 227);
	const float mean_vals[3] = { 104.f, 117.f, 123.f };
	in.substract_mean_normalize(mean_vals, 0);

	ncnn::Extractor ex = squeezenet.create_extractor();
	ex.input("data", in);
	ncnn::Mat out;
	ex.extract("prob", out);

	cls_scores.resize(out.w);
	for (int j = 0; j < out.w; j++) {
		cls_scores[j] = out[j];
	}

	return 0;
}

static int print_topk(const std::vector<float>& cls_scores, int topk)
{
	int size = cls_scores.size();
	std::vector<std::pair<float, int> > vec;
	vec.resize(size);
	for (int i = 0; i < size; i++) {
		vec[i] = std::make_pair(cls_scores[i], i);
	}

	std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(), std::greater<std::pair<float, int> >());

	for (int i = 0; i < topk; i++) {
		float score = vec[i].first;
		int index = vec[i].second;
		fprintf(stderr, "%d = %f\n", index, score);
	}

	return 0;
}*/

static int load_movenet(char* model) {

	int modelid = 0;// 默认使用lightning
	int cpugpu = 1;// 默认使用cpu
	// 不是lightning的话就使用thunder
	if (strcmp(model, "lightning") != 0) modelid = 1;

	if (modelid < 0 || modelid > 6 || cpugpu < 0 || cpugpu > 1)
	{
		return -1;
	}

	const char* modeltypes[] =
	{
		"lightning",
		"thunder",
	};

	const int target_sizes[] =
	{
		192,
		256,
	};

	const float mean_vals[][3] =
	{
		{ 127.5f, 127.5f,  127.5f },
		{ 127.5f, 127.5f,  127.5f },
	};

	const float norm_vals[][3] =
	{
		{ 1 / 127.5f, 1 / 127.5f, 1 / 127.5f },
		{ 1 / 127.5f, 1 / 127.5f, 1 / 127.5f },
	};

	const char* modeltype = modeltypes[(int)modelid];
	int target_size = target_sizes[(int)modelid];
	bool use_gpu = (int)cpugpu == 1;

	// reload
	{
		if (!g_nanodet) g_nanodet = new NanoDet;
		g_nanodet->load(modeltype, target_size, mean_vals[(int)modelid], norm_vals[(int)modelid], use_gpu);
	}
	return 0;
}

static void on_image_render(cv::Mat& rgb)
{
	// nanodet
	{
		// ncnn::MutexLockGuard g(lock);

		if (g_nanodet)
		{
			std::vector<keypoint> points;
			g_nanodet->detect_point(rgb, points);
			if(isDrawLine) g_nanodet->draw(rgb, points);

			// 改用tjs表达式的形式执行
			string s = "[";
			for (int i = 0; i < points.size(); i++) {
				if (i < points.size() - 1) s = s + "%[\"x\"=>" + to_string(points[i].x) + ",\"y\"=>" + to_string(points[i].y) + ",\"score\"=>" + to_string(points[i].score) + "],";
				else if (i == points.size() - 1) s = s + "%[\"x\"=>" + to_string(points[i].x) + ",\"y\"=>" + to_string(points[i].y) + ",\"score\"=>" + to_string(points[i].score) + "]]";
			}
			//TVPAddLog("Execute tjs Script:");
			string code_s = "if(bodyAI.onFrame!=void){ bodyAI.onFrame(" + s + ");}";
			//string code_s = "bodyAI.onFrame(" + s + ");";
			const char *p = (char*)code_s.data();
			//ttstr code(p);
			TVPExecuteScript(p);

			//std::vector<tjs_keypoint> tjs_body;
			/*string s = "[";
			for (int i = 0; i < points.size(); i++) {
				if(i < points.size() - 1) s = s + "{\"x\":" + to_string(points[i].x) + ",\"y\":" + to_string(points[i].y) + ",\"score\":" + to_string(points[i].score) + "},";
				else if(i == points.size() - 1) s = s + "{\"x\":" + to_string(points[i].x) + ",\"y\":" + to_string(points[i].y) + ",\"score\":" + to_string(points[i].score) + "}]";
			}

			tTJSVariant _result;
			globalBean->PropGet(0, L"bodyAI", NULL, &_result, globalBean);
			const tjs_char *method = TJS_W("onFrame");
			tTJSVariant result;
			tTJSVariantClosure bodyObj = _result.AsObjectClosure();
			const char *p = (char*)s.data();
			tjs_char *body_string = trstring2wchar(p);
			tTJSVariant temp = body_string;
			tTJSVariant *param[1] = { &temp };
			bodyObj.FuncCall(0, method, NULL, &result, 1, param, globalBean);*/
			// bodyObj.Release();
		}
		else
		{
			//draw_unsupported(rgb);
		}
	}
	// draw_fps(rgb);
}

DWORD WINAPI cameraThreadFunc(LPVOID param) {

	//打开第一个摄像头
	VideoCapture cap(0);
	//判断摄像头是否打开
	if (!cap.isOpened())
	{
		TVPAddLog("摄像头未成功打开");
	}

	bool *contents = (bool*)param;
	if (contents) {
		//创建窗口
		namedWindow("krkr摄像头窗口", 1);
	}

	while (1)
	{
		//创建Mat对象
		Mat frame;
		//从cap中读取一帧存到frame中
		bool res = cap.read(frame);
		if (!res)
		{
			break;
		}

		//判断是否读取到
		if (frame.empty())
		{
			break;
		}
		// 帧回调函数
		on_image_render(frame);
		if (contents) {
			//显示摄像头读取到的图像
			imshow("krkr摄像头窗口", frame);
		}
		// 这里wait一下，不然窗口会卡死
		waitKey(1);
	}

	cap.release();
	if (contents) {
		destroyAllWindows();
	}
	return 0L;
}

class bodyAI {
private:
	
public:
	// 初始化
	static tjs_error TJS_INTF_METHOD init(
		tTJSVariant *result,
		tjs_int numparams,
		tTJSVariant **param,
		iTJSDispatch2 *objthis)
	{
		if (numparams != 2) return TJS_E_BADPARAMCOUNT;//必传2个参数，一个是NCNN的模型，一个是tensorflow分类器模型
		// ncnn可以不带后缀
		char* ncnn_model = wchar2char(param[0]->GetString());
		// tensorflow需要带上后缀
		char* body_model = wchar2char(param[1]->GetString());
		load_movenet(ncnn_model);
		// 绑定tjs全局对象
		globalBean = objthis;
		//tTJSVariant _result;
		//objthis->PropGet(0, L"bodyAI", NULL, &_result, objthis);
		//tjsBodyAIClass = &_result;
		//initModel(body_model);
		return TJS_S_OK;
	}

	// 设置模型的参数（unuse）
	static tjs_error TJS_INTF_METHOD setModelParam(
		tTJSVariant *result,
		tjs_int numparams,
		tTJSVariant **param,
		iTJSDispatch2 *objthis)
	{
		if (numparams != 2) return TJS_E_BADPARAMCOUNT;//要传入身体关键点的个数和模型的分类个数
		bodyPointNum = param[0]->AsInteger();
		bodyClassNum = param[1]->AsInteger();
		return TJS_S_OK;
	}

	// 打开摄像头
	static tjs_error TJS_INTF_METHOD openCamera(
		tTJSVariant *result,
		tjs_int numparams,
		tTJSVariant **param,
		iTJSDispatch2 *objthis)
	{
		if (numparams != 2) return TJS_E_BADPARAMCOUNT;//要传两个参数，摄像头画面是否显示，人体轮廊是否显示
		// 摄像头画面窗口
		bool isOpenWin = param[0]->AsInteger();
		// 人体轮廊
		isDrawLine = param[1]->AsInteger();
		/*thread t1(cameraThread2, true);
		t1.join();*/
		DWORD dwThreadID = 0;
		HANDLE cameraHandle = CreateThread(NULL, NULL, cameraThreadFunc, &isOpenWin, 0, 0);
		CloseHandle(cameraHandle);
		Sleep(1);
		return TJS_S_OK;
	}

	// 注册帧回调函数（unuse）
	static tjs_error TJS_INTF_METHOD registerFrame(
		tTJSVariant *result,
		tjs_int numparams,
		tTJSVariant **param,
		iTJSDispatch2 *objthis)
	{
		if (numparams != 1) return TJS_E_BADPARAMCOUNT;//要传入一个帧回调函数，每帧调用
		frameCallBack = param[0]->AsObject();
		return TJS_S_OK;
	}
};

NCB_REGISTER_CLASS(bodyAI) {
	RawCallback("init", &Class::init, 0);
	RawCallback("openCamera", &Class::openCamera, 0);
	//RawCallback("setModelParam", &Class::setModelParam, 0);
	//RawCallback("registerFrame", &Class::registerFrame, 0);
}
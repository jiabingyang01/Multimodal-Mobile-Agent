![](assets/logo.jpg?v=1&type=image)

## Multimodal-Mobile-Agent: 通过多代理协作有效导航的多模态移动设备操作助手

本项目基于[Mobile-Agent-v2](https://github.com/X-PLUG/MobileAgent/tree/main/Mobile-Agent-v2)开发

## 📋介绍
### 背景
移动设备操作任务正日益成为一种流行的多模态人工智能应用场景。目前的多模态大语言模型(MLLMs)受其训练数据的限制，很难发挥其作为操作助手的能力。已有的工作如Mobile-Agent通过利用单agent模型使用视觉感知方案识别手机屏幕截图上的图标，再根据大语言模型的理解能力进行动作推理，实现了根据用户指令自动操作移动手机。但是随着历史序列的不断增长，单agent的能力有所不足，在处理复杂操作指令时很难满足用户需求。多agent作为最近兴起的一项技术，通过设计多个智能体使它们互相交互，以模拟真实世界人类合作，在各个领域展现出广阔的前景。斯坦福小镇等就是利用多agent模拟人类世界行为。Moile-agent-v2作为是Mobile-agent的改进版，利用多agent来提升性能。但是它只能支持文本指令输入，现实中人们往往需要利用语音等模态输入，在Mobile-Agent-v2的基础上，我进行了相应改进，使之支持文本，图像和语音输入，并将Agent智能体的基础模型从GPT-4V换为GPT-4o，实现了一个支持多模态输入的多智能体协作移动手机操作助手Multimodal-Mobile-Agent。

###  原理

Multimodal-Mobile-Agent有3个专门的Agent角色：规划代理(Planning Agent)、决策代理(Decision Agent)和反思代理(Reflection Agent)。还包括视觉感知模块和记忆单元，以增强智能体的屏幕识别能力和从历史中导航焦点内容的能力。首先，Planning Agent更新任务进度，允许决策代理导航当前任务的进度。然后Decision Agent根据当前任务进度、当前屏幕状态和反射(如果最后一个操作是错误的)进行操作。随后，Reflection Agent观察操作前后的屏幕，以确定操作是否符合预期。另外通过引入GPT-4o模型和Whisper模型使之支持文本、图像和语音三种模态的指令输入。

![](assets/role.jpg?v=1&type=image)

#### 1. 视觉感知模块(Visual Perception Module)

即使是最先进的MLLM，在端到端处理时，屏幕识别仍然具有挑战性。因此，我们加入了视觉感知模块来增强屏幕识别能力。在该模块中，我们使用了三个工具：文本识别工具、图标识别工具和图标描述。在该模块中输入截图，最终将产生屏幕上呈现的文本和图标信息，以及它们各自的坐标。对于文本识别工具，我们使用了ModelScope3中的文档OCR识别模型ConvNextViT-document。对于图标识别工具，我们采用了GroundingDINO，一种能够基于自然语言提示检测物体的检测模型。对于图标描述工具，我们使用了Qwen-VL-plus。这个过程用下面的公式表示：
$$
\begin{equation}
P_t = VPM(S_t)
\end{equation}
$$

其中$P_t$表示第$t$次迭代中屏幕的感知结果。

#### 2. 内存单元(Memory Unit)

由于计划代理产生的任务进度是文本形式的，从历史屏幕中导航焦点内容仍然具有挑战性。为了解决这个问题，我们设计了一个内存单元来存储历史屏幕中与当前任务相关的焦点内容。内存单元作为短时记忆模块，随着任务的进行而更新。对于包含多个应用的场景，内存单元至关重要。例如，如图1所示，决策代理观察到的天气信息将用于后续操作。此时，与天气APP页面相关的信息将在内存单元中更新。


#### 3. 规划代理(Planning Agent)

规划代理的作用是总结历史操作和跟踪任务进展。我们将决策主体在第t次迭代时产生的操作定义为$O_t$。在决策Agent做出决策之前，规划Agent从上一次迭代开始观察决策Agent的操作$O_{t-1}$，并将任务进度 $TP_{t-1}$ 更新为$TP_t$。任务进度包括已经完成的子任务。在生成任务进度后，规划Agent将其传递给决策Agent。这有助于决策Agent考虑尚未完成的任务内容，从而有利于下一步操作的生成。如图1所示，规划智能体的输入由4部分组成：用户指令Ins、记忆单元中的焦点内容$FC_t$、先前的操作$O_{t-1}$和先前的任务进度$TP_{t - 1}$。基于上述信息，规划Agent生成$TP_t$。这个过程用下面的公式表示：
$$
\begin{equation}
    TP_t = PA(Ins, O_{t-1}, TP_{t-1}, FC_{t-1})
\end{equation}
$$
其中$PA$表示规划Agent使用的LLM，项目中使用的是GPT-4o。

#### 4. 决策代理(Decision Agent)
决策代理在决策阶段进行操作，生成操作$O$并在设备上实现，同时负责更新记忆单元中的焦点内容$FC$。这一过程在图1所示的决策阶段中进行了说明，并由下面的公式表示：
$$
\begin{equation}
    O_t = DA(Ins, TP_{t-1}, FC_{t-1}, R_{t-1}, S_t, P_t)
\end{equation}
$$
其中，$DA$表示决策智能体的MLLM，项目中使用的是GPT-4o，Rt表示反射智能体的反射结果。

**操作空间** 为了降低操作的复杂度，我们设计了一个操作空间，并限制了决策Agent只能从这个空间中选择操作。对于具有更高自由度的操作，例如敲击和刷新，我们纳入了一个额外的参数空间来定位或处理特定的内容。下面是对操作空间的详细描述：
* Open app (appname). If the current page is the home page, this operation can be used to open the app named "app name". 
* Tap $(x, y)$. This operation is used to tap on the position with coordinates $(x, y)$. 
* Swipe $(x1, y1), (x2, y2)$. This operation is used to swipe from the position with coordinates $(x1, y1)$ to the position with coordinates $(x2, y2)$. 
* Type (text). If the current keyboard is in an active state, this operation can be used to input the content of "text" in the input box.
* Home. This operation is used to return to the home page from any page.
* Stop. If the decision agent thinks that all requirements have been fulfilled, it can use this operation to terminate the entire operation process.

**记忆单元更新** 由于决策主体所做的每一个操作都具有高度的任务相关性，并且基于当前页面的视觉感知结果，因此非常适合在屏幕页面内观察与任务相关的焦点内容。据此，我们赋予了决策主体更新记忆单元的能力。在进行决策时，提示决策代理观察当前屏幕页面内是否存在与任务相关的焦点内容。如果观察到这样的信息，决策主体将其更新在内存中，以便在后续的决策中参考。这个过程用下面的公式表示：
$$
\begin{equation}
    FC_t = DA(Ins, FC_{t-1}, S_t, P_t)
\end{equation}
$$

#### 5. 反思代理(Reflection Agent)
即使有了视觉感知模块，Multimodal-Mobile-Agent仍然可能产生意想不到的操作。在某些特定的情景下，MLLMs甚至会产生严重的幻觉。因此，我们设计了反射代理来观察一个决策代理操作前后的屏幕状态，以确定当前操作是否符合预期。这个过程用下面的公式表示：
$$
\begin{equation}
    R_t = RA(Ins, FC_{t}, O_t, S_t, P_t, S_{t+1}, P_{t+1})
\end{equation}
$$
其中$RA$表示反思代理使用的MLLM，项目中使用的是GPT-4o。
如图1所示，反射代理在操作执行后会产生三种类型的反射结果：错误操作、无效操作和正确操作。下面将对这三种反射结果进行描述：
* 错误操作是指导致设备进入与任务无关的页面的操作。例如，代理人打算在短信APP中与联系人A聊天，但却意外打开了联系人B的聊天页面。
* 无效操作是指不导致当前页面任何变化的操作。例如，智能体打算点击一个图标，但它却点击了图标旁边的空白处。
* 正确操作是指符合决策智能体预期的操作，是实现用户指令要求的一个步骤。

如果操作出错，页面将恢复到操作前的状态。如果操作无效，页面将保持当前状态。在操作历史中既不记录错误的操作，也不记录无效的操作，以防止代理人跟随这些操作。如果操作正确，则在操作历史中更新操作，并将页面更新到当前状态。

#### 6. 多模态输入(Multimodal Input)
由于在Mobile-Agent-v2中只支持文本输入，为了能使该助手使用起来更为便利，我们扩展了模态输入方式。文本输入只需常用的大语言模型如GPT-4o即可，图像输入我们使用了GPT-4o模型，因为它在很多多模态任务上都达到了SOTA效果。我们可以通过输入一张图片和提示，让模型根据图片和我们的提示去完成手机操作。如给一张科比的图片，让模型去查找有关图片中的人的信息。对于语音输入，由于openai的Whisper-1模型无法通过中转api调用，不知道什么原因，我便使用了可以部署在本地的openai的whisper(large)模型。这是一个能够实现语音和文本相互转换的大语言模型，使用前需要下载到本地。这样我们可以将用户语音指令转换为文本指令并让多个代理进行操作实现。通过多模态输入的实现，增加了该语音助手的可用性。

#### 7. Summary

* 支持文本、图像和语音多模态输入。
* 一个用于解决在长上下文图文交错输入中导航的多智能体架构。
* 增强的视觉感知模块，用于提升操作准确率。
* 凭借GPT-4o进一步提升操作性能和速度。

## 🔧开始

❗目前仅安卓和鸿蒙系统（版本号 <= 4）支持工具调试。其他系统如iOS暂时不支持使用Mobile-Agent。

### 安装依赖
```
pip install -r requirements.txt
```

### 准备通过ADB连接你的移动设备

1. 下载 [Android Debug Bridge](https://developer.android.com/tools/releases/platform-tools?hl=en)（ADB）。
2. 在你的移动设备上开启“USB调试”或“ADB调试”，它通常需要打开开发者选项并在其中开启，具体如何打开可参照百度。
3. 通过数据线连接移动设备和电脑，在手机的连接选项中选择“传输文件”。
4. 用下面的命令来测试你的连接是否成功: ```/path/to/adb devices```。如果输出的结果显示你的设备列表不为空，则说明连接成功。
5. 如果你是用的是MacOS或者Linux，请先为 ADB 开启权限: ```sudo chmod +x /path/to/adb```。
6.  ```/path/to/adb```在Windows电脑上将是```xx/xx/adb.exe```的文件格式，而在MacOS或者Linux则是```xx/xx/adb```的文件格式。

### 在你的移动设备上安装 ADB 键盘
1. 下载 ADB 键盘的 [apk](https://github.com/senzhk/ADBKeyBoard/blob/master/ADBKeyboard.apk)  安装包。
2. 在设备上点击该 apk 来安装。
3. 在系统设置中将默认输入法切换为 “ADB Keyboard”。

### 选择适合的运行方式

1.在 ```run.py``` 的31行起编辑你的设置， 并且输入你的 ADB 路径，指令，GPT-4o API URL 和 Token(API-key)。

2.通过设置choice选择你需要的使用的指令模态：
* choice=1代表使用文本指令，只需修改```run.py```57行的```Instruction```
* choice=2代表输入图像和文本prompt作为指令，此时需将修改```run.py```60行的```image_path```设为你要输入的图像的地址，同时修改61行的```prompt```以指导助手根据图片进行的操作。
* choice=3代表使用语音指令，只需修改```run.py```71行的```audio_path```设为你要输入的音频的地址。

2.选择适合你的设备的图标描述模型的调用方法：
  - 如果您的设备配备了高性能GPU，我们建议使用“local”方法。它是指在本地设备中部署图标描述模型。如果您的设备足够强大，则通常具有更好的效率。
  - 如果您的设备不足以运行7B 大小的 LLM，请选择“ api”方法。我们使用并行调用来确保效率。

3.选择图标描述模型：
  - 如果选择“local”方法，则需要在“qwen-vl-chat”和“qwen-vl-chat-int4”之间进行选择，其中“qwen-vl-chat”需要更多的GPU内存，但提供了更好的性能与“qwen-vl-chat-int4”相比。同时，“qwen_api”可以是空置的。
  - 如果您选择“api”方法，则需要在“qwen-vl-plus”和“qwen-vl-max”之间进行选择，其中“qwen-vl-max”需要更多的费用，但与“qwen-vl-plus”相比提供了更好的性能。此外，您还需要申请[Qwen-VL 的 API-KEY](https://help.aliyun.com/document_detail/2712195.html?spm=a2c4g.2712569.0.0.5d9e730aymB3jH)，并将其输入到“qwen_api”。

4.您可以在“add_info”中添加操作知识（例如，完成您需要的指令所需的特定步骤），以帮助更准确地运行移动设备。

5.如果您想进一步提高移动设备的效率，则可以将“ reflection_Switch”和“ memory_switch”设置为“ False”。
  - “ reflection_switch”用于确定是否在此过程中添加“反思智能体”。这可能会导致操作陷入死周期。但是您可以将操作知识添加到“ add_info”中以避免它。
  - “ memory_switch”用于决定是否将“内存单元”添加到该过程中。如果你的指令中不需要在后续操作中使用之前屏幕中的信息，则可以将其关闭。

### 运行
```
python run.py
```

## 📺演示
### 1.文本输入
我们设置```choice = 1```，使模型接受文本输入，我们输入文本指令```instruction = "打开百度地图导航到北京大学"```，运行结果见text_instruction.mp4。可以看到Mutilmodal-Mobile-Agent成功操作手机打开了百度地图并搜索了北京大学随后导航去北京大学。

#### text.mp4

<video src="https://1327711314.vod-qcloud.com/8760b000vodcq1327711314/097589d81253642700215380873/AHXKWrNaPVEA.mp4" controls="controls" width="800" height="600">
</video>

### 2.图像输入
我们设置```choice = 2```，使模型接受图像输入，我们输入一张北京大学的图像，如下，让模型理解这个图像所表达的位置，然后利用文本prompt得到响应操作指令，```prompt = "I want to go to the place in this image. What should I do with a mobile phone."```，GPT-4o根据图片和prompt得到的指令为```Instruction = "Open Baidu map and go to Beijing University"```，可见成功理解了用户的需求，运行结果见image.mp4。

![](file/image/1.jpg?v=1&type=image)

#### image.mp4

<video src="https://1327711314.vod-qcloud.com/8760b000vodcq1327711314/9c5900f11253642700217071921/b7rL7mO0bjwA.mp4" controls="controls" width="800" height="600">
</video>

### 3.语音输入
我们设置```choice = 3```，输入下面的音频，whisper模型将其转译为```"Open Baidu map and go to Beijing University"```，将其作为操作指令，运行结果见image.mp4，可以看到虽然指令中“导航”并没有翻译对，但助手最终仍然成功完成了任务。另外在其中我们可以看到助手有一步点击搜索框时不小心点到了旁边的语音助手图标，导致出现无效操作，经过反思后重新生成了正确的动作，可见反思智能体在纠正操作错误方面的有效性。

<audio src="./file/audio/1.MP3" autoplay="true" controls="controls" width="800" height="600">
</audio>

#### audio.mp4

<video src="https://1327711314.vod-qcloud.com/8760b000vodcq1327711314/1298b4be1253642700215759194/h3mhSfdqerEA.mp4" controls="controls" width="800" height="600">
</video>

**注**：如果卡顿请点击链接观看

## 总结
### 优点
本次项目设计的Mutilmodal-Mobile-Agent通过设计三个Agent并使它们交互协作，能够成功的理解并完成用户的多模态操作作令，包括文本，图像和语音，进一步扩展了它在移动设备操作领域的可用性。另外我们使用GPT-4o作为三个智能体的基础模型，大大增强了智能体的对用户指令的理解和推理能力，也让该助手更为智能和准确。反思智能体的引入也使得模型能够及时纠正错误最终完成任务。综合来看，Mutilmodal-Mobile-Agent是一个比较具有实际应用价值的多模态移动手机操作助手。

### 缺点
尽管Mutilmodal-Mobile-Agent可以以较高的准确率完成用户的操作任务，但受限于大模型的推理速度和API的调用限制，整个操作的完成时间较长，很难满足快速响应的需求。对比于现在各个公司的语音助手，其还有待进一步的改进。在面对复杂操作时，Mutilmodal-Mobile-Agent往往很难胜任，这既受到较长的历史操作序列的影响，又受制于大模型的性能瓶颈，因此在未来大模型性能进一步提高的情况下，该助手可能会更加适应用户的需求。



## 📑参考

* [Mobile-agent:Mobile-Agent: Autonomous Multi-Modal Mobile Device Agent with Visual
Perception](https://arxiv.org/pdf/2401.16158)

* [Mobile-Agent-v2: Mobile Device Operation Assistant with Effective Navigation via Multi-Agent Collaboration](https://arxiv.org/pdf/2406.01014)

* https://github.com/X-PLUG/MobileAgent/tree/main

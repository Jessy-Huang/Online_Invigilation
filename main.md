## 一、项目背景介绍
疫情之后，线上考试的需求剧增，由此而来的线上监考的所需花费的人力和物力也不断增加。本项目基于2020年东华大学线上考试收集的监考画面，使用paddleclda实现自动监考

## 二、数据介绍
本项目用的是大学生线上考试监考监控数据集，一共有六类，分别是：qualifIed、more_than_one_people、uncorrelated、using_phone、without_computer、without_examinee、wrong_angel共有500+张图片，图片数量比较多，加载时间比较长，请小伙伴们耐心等待
* 需要注意的是：
    * 数据集存放路径要与配置文件中一直，不要忘记修改
    * 数据列表文件中路径与标签之间的空格要划分准确
    * 数据列表文件中不要包含其他文件

## 三、模型介绍
图像分类是根据图像的语义信息将不同类别图像区分开来，是计算机视觉中重要的基础任务。
  图像分类在很多领域有广泛应用，包括零售商品分类、农作物品质分级、医学领域的图像识别、交通标志分类等。作为计算机视觉任务的基础，还可辅助目标检测、图像分割、物体跟踪、行为分析等其他高层视觉任务组网和效果提升，比如经典的目标检测模型FasterRCNN的结构，骨干网络即使用分类任务中的网络结构，如ResNet。作为骨干网络，分类任务的预训练模型也可以帮助其他视觉任务在训练时更快地收敛。
 ### ResNet介绍 ###
 ![](https://ai-studio-static-online.cdn.bcebos.com/ed626966db0d4c3c90191c7290d13062bcc3f07fee6a4f66ae6bbf310857bc4e)
 上面这张图其实也就是给出了ILSVRC图像分类数据集的top-1 error指标，resnet出来时，远超其他模型的结果。
** 下面给出了ResNet的核心结构：残差模块**
![](https://ai-studio-static-online.cdn.bcebos.com/97a0360b437747b2b694e042e6dc7e8da7e90680f48b42129a0f6ea963f2cdd1)
在残差结构中，一个支路经过各种卷积运行，另一个支路直接连接到输出，这两个支路相加之后得到输出，相当于卷积计算的支路只需要计算残差项，这大大降低了模型训练过程中的学习难度。下面也给出了ResNet18的结构。
![](https://ai-studio-static-online.cdn.bcebos.com/9c53878b672d47a28622125f8904e47a32a3a3e488c246b280a34ceda714414e)
上面是标准的ResNet结构，李沐大神及其团队在后来对ResNet做了一系列的改进。下面给出了最左边是最开始的ResNet-Va结构，Vb对这个左边的特征变换通路的降采样卷积做了调整，把降采样的步骤从最开始的第一个1x1卷积调整到中间的3x3卷积中；Vc结构则是将最开始这个7x7的卷积变成3个3x3的卷积，在感受野不变的情况下减少了存储；而Vd是修改了降采样残差模块右边的特征通路。把降采样的过程由平均池化这个操作去替代了，这一系列的改进，几乎没有带来新增的预测耗时，结合适当的训练策略，比如说标签平滑以及mixup这种数据增广方式，精度可以提升高达2.5%。
![](https://ai-studio-static-online.cdn.bcebos.com/da5c0b8a31304bc1a45389fb5d8b86398363fe90ab2f492aa74a680937c8e4f3)


## 四、模型训练
### 1 数据处理 ##
```Python
#解压数据集
!unzip -oq data/data128035/dataset.zip
!mv dataset PaddleClas/dataset


from PIL import Image

# 读取图片
png_img = Image.open('PaddleClas/dataset/dataset/qualified/18.jpg')
png_img  # 展示图片
```
得到的结果如下
![](https://ai-studio-static-online.cdn.bcebos.com/55ef44d1894544a78b71faf6b99d481036c1d63de40146ddb3b2d1e215fd88b0)


在训练之前需要将数据集分为训练集和测试集，并且需要转换为txt格式，因此做如下处理

```Python
#划分数据集
import codecs
import os
import random
import shutil
from PIL import Image

train_ratio = 4.0 / 5
#all_file_dir = 'PaddleClas/dataset/fer62013'
all_file_dir = 'PaddleClas/dataset/dataset'
class_list = [c for c in os.listdir(all_file_dir) if os.path.isdir(os.path.join(all_file_dir, c)) and not c.endswith('Set') and not c.startswith('.')]
class_list.sort()
print(class_list)
train_image_dir = os.path.join(all_file_dir, "trainImageSet")
if not os.path.exists(train_image_dir):
    os.makedirs(train_image_dir)
    
eval_image_dir = os.path.join(all_file_dir, "evalImageSet")
if not os.path.exists(eval_image_dir):
    os.makedirs(eval_image_dir)

train_file = codecs.open(os.path.join(all_file_dir, "train.txt"), 'w')
eval_file = codecs.open(os.path.join(all_file_dir, "eval.txt"), 'w')

with codecs.open(os.path.join(all_file_dir, "label_list.txt"), "w") as label_list:
    label_id = 0
    for class_dir in class_list:
        label_list.write("{0}\t{1}\n".format(label_id, class_dir))
        image_path_pre = os.path.join(all_file_dir, class_dir)
        for file in os.listdir(image_path_pre):
            try:
                img = Image.open(os.path.join(image_path_pre, file))
                if random.uniform(0, 1) <= train_ratio:
                    shutil.copyfile(os.path.join(image_path_pre, file), os.path.join(train_image_dir, file))
                    train_file.write("{0} {1}\n".format(os.path.join("trainImageSet", file), label_id))
                else:
                    shutil.copyfile(os.path.join(image_path_pre, file), os.path.join(eval_image_dir, file))
                    eval_file.write("{0} {1}\n".format(os.path.join("evalImageSet", file), label_id))
            except Exception as e:
                pass
                # 存在一些文件打不开，此处需要稍作清洗
        label_id += 1
            
train_file.close()
eval_file.close()
```

输出为：
['more_than_one_people', 'qualified', 'uncorrelated', 'using_phone', 'without_computer', 'without_examinee', 'wrong_angle']
 并且得到了train.txt、eval.txt和label_list.txt三个文件

 **查看划分的情况**

```python
 !tree -d PaddleClas/dataset
```
 PaddleClas/dataset
└── dataset
    ├── evalImageSet
    ├── more_than_one_people
    ├── qualified
    ├── trainImageSet
    ├── uncorrelated
    ├── using_phone
    ├── without_computer
    ├── without_examinee
    └── wrong_angle

```python
! head PaddleClas/dataset/dataset/train.txt
```
trainImageSet/26.jpg 0
trainImageSet/29.jpg 0
trainImageSet/1.jpg 0
trainImageSet/21.jpg 0
trainImageSet/34.jpg 0
trainImageSet/16.jpg 0
trainImageSet/118.jpg 0
trainImageSet/4.jpg 0
trainImageSet/13.jpg 0
trainImageSet/5.jpg 0

### 2 文件配置 ###
**学习率与优化配置**
```python
LEARNING_RATE:
    function: 'Cosine' decay方法名  ["Linear", "Cosine","Piecewise", "CosineWarmup"]         
    params: 初始学习率     大部分的神经网络选择的初始学习率为0.1，batch_size是256，所以根据实际的模型大小和显存情况，可以将学习率设置为0.1*k,batch_size设置为256*k              
        lr: 0.1   
*还可设置的参数
params:
	decayepochs	 piecewisedecay中衰减学习率的milestone
params:
	gamma	    piecewisedecay中gamma值	
params:
	warmupepoch	 warmup轮数	
parmas:
	steps	    lineardecay衰减steps数	
params:
	endlr	    lineardecayendlr值	

OPTIMIZER:
    function: 'Momentum' 优化器方法名 ["Momentum", "RmsProp"]
    params:
        momentum: 0.9 momentum值
    regularizer:
        function: 'L2' 正则化方法名	
        factor: 0.000070 正则化系数
```
**训练配置**
```python
TRAIN:
    batch_size: 32 批大小
    num_workers: 4 数据读取器worker数量
    file_list: "./dataset/NEU-CLS/train.txt" train文件列表
    data_dir: "./dataset/NEU-CLS" train文件路径
    shuffle_seed: 0 用来进行shuffle的seed值
    transforms: 数据处理
        - DecodeImage:
            to_rgb: True 数据转RGB
            to_np: False 数据转numpy
            channel_first: False 按CHW排列的图片数据
        - RandCropImage: 随机裁剪
            size: 224
        - RandFlipImage: 随机翻转
            flip_code: 1
        - NormalizeImage:
            scale: 1./255. 归一化scale值
            mean: [0.485, 0.456, 0.406] 归一化均值
            std: [0.229, 0.224, 0.225] 归一化方差
            order: '' 归一化顺序
        - ToCHWImage: 调整为CHW
    mix:                       
        - MixupOperator:    
            alpha: 0.2      
*还可设置的参数
-CropImage	
	size:	裁剪大小
-ResizeImage	
	resize_short:	按短边调整大小

```
**测试配置**
```python
VALID:
    batch_size: 64
    num_workers: 4
    file_list: "./dataset/NEU-CLS/eval.txt"
    data_dir: "./dataset/NEU-CLS"
    shuffle_seed: 0
    transforms:
        - DecodeImage:
            to_rgb: True
            to_np: False
            channel_first: False
        - ResizeImage:
            resize_short: 256
        - CropImage:
            size: 224
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
        - ToCHWImage:

```


### 3 模型训练 ###
```Python
#设置环境变量
%cd PaddleClas
import os 
os.environ['PYTHONPATH']="/home/aistudio/PaddleClas"

#加载预训练模型
!python ../download_model.py ResNet50_vd_pretrained
!mv ../ResNet50_vd_pretrained ./

#开始训练
!python -m paddle.distributed.launch --selected_gpus="0"  tools/train.py -c ../fer.yaml -o pretrained_model=./ResNet50_vd_pretrained
```


## 五、模型评估
### 1 理论评估 ###
```Python
!python -m paddle.distributed.launch --selected_gpus="0" tools/eval.py \
    -c ../eval.yaml \
    -o pretrained_model=output/ResNet50_vd/best_model/ppcls
```

**结果展示**
```
2020-08-03 23:58:36 INFO: eval step:0    loss:  0.4387 top1: 0.8906 top5: 1.0000 elapse: 0.690s
2020-08-03 23:58:37 INFO: END eval loss_avg:  0.6619 top1_avg: 0.9000 top5_avg: 0.9969 elapse_sum: 1.102ss
INFO 2020-08-03 15:58:39,735 launch.py:223] Local procs complete, POD info:rank:0 id:None addr:127.0.0.1 port:None visible_gpu:[] trainers:["gpu:['0'] endpoint:127.0.0.1:56541 rank:0"]
```
上面展示的结果是笔者训练201个epochs，在测试集上top1的准确率为0.9000，经过测试在进行400次迭代之后会达到100%的准确率。

### 2 运用检验 ###
为了模拟真实性，我们直接从手机相册里拿出来一张隔壁老王的照片试试，放在PaddleClas/test.PNG中
```Python
!python tools/export_model.py \
    --model='ResNet50_vd' \
    --pretrained_model=output/ResNet50_vd/best_model/ppcls \
    --output_path=./inference


!python tools/infer/predict.py \
    -m inference/model \
    -p inference/params \
    -i "test1.jpg" \
    --use_gpu=1

```

测试照片
![](https://ai-studio-static-online.cdn.bcebos.com/fea12d536ab74c7e8ce4d47bd468a9085ab6970e87d04cfb95d79e2daab6747c)


2022-02-23 23:00:08,978-INFO: class: 2
2022-02-23 23:00:08,979-INFO: score: 0.7071830034255981

['more_than_one_people', 'uncorrelated', 'qualified', 'using_phone', 'without_computer', 'without_examinee', 'wrong_angle']

可以看到预测正确，图片的标签为2，与label.txt对应 qualified



## 六、总结与升华
该项目是基于PaddleClas的一个模型，虽然得到了结果，但是并没有自己构建模型，后期希望自己搭建模型来得到一个好的训练结果。
该项目的有点是数据集是自己采集和整理，根据疫情期间的监控画面得到的数据集

## 七、个人总结
作者是上海某211大学大四自动化系学生，研究方向为优化算法和视觉处理

## 提交链接
aistudio链接：https://aistudio.baidu.com/aistudio/projectdetail/3528958

github链接：https://github.com/Jessy-Huang/Online_Invigilation

gitee链接：https://gitee.com/huang-jiacui/online-invigilation-model

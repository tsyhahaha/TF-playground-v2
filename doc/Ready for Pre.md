# Ready for Pre

## O、技术积淀

TSY：

* 参与开发名为《走马Go》的校园体育课程网站，基于 vue.js，负责大部分的前端开发。
* 自学部分 cs231 和 cs229 网课，有较为完整的机器学习基础

XZQ：

* 来自人工智能学院，系统学习过人工智能相关课程（机器学习，深度学习，RL等）。
* 有开发大项目的经验，并在本学期选修数据可视化课程 


## 一、app部分实现原理

### 1、项目结构

```cmd
index.html	# 网页结构实现
styles.css	# CSS 样式文件
analytics.js  # Google Analytics 部署
src
|-dataset.ts	# 数据集生成
|-heatmap.ts	# heatmap 类的定义，用于绘制 result 及 node 的热图
|-linechart.ts	# linechart 类的定义，用于绘制 OUTPUT 的 LOSS 曲线
|-nn.ts		    # FC 计算模块定义
|-playground.ts	# 主页面交互操作实现
|-seedrandom.ts
|-state.ts		# 页面参数解析、存储相关
```

更具体一些的，做了个 MindX 思维导图（不全）放下面了，可能可以拆分一下放在 ppt 里讲一下我们对各个文件功能的理解。

### 2、网页主要功能的实现

需要注重执行逻辑层面的东西，这个还得细看 playground.ts ，属实有些复杂，如果可以考虑形成流程图。

### 3、网络结构搭建原理

## 二、app添加功能设计

### 1、UI 草图

（先放着占个坑，后期需要做出有关新功能的初步草图设计）

![image-20230302215913811](https://20220923img.oss-cn-hangzhou.aliyuncs.com/markdown/image-20230302215913811.png)

### 2、功能实现设计

> 这部分感觉要突出这两类东西的功能，突出**加上和没加上的区别**，可能需要添加一些记录用的参数，譬如收敛时间等

#### Batch Normalization / Layer Normalization

这部分应该作为一个参数选择项

#### Adam

### 3、其他的想法

（搞点儿新奇的想法？增加点竞争力）设计上的，原理上的，实现上的，可优化的点。不过这部分并不是很重要，因为得分是速度优先。

## 三、后期规划

感觉没啥规划？每周看情况安排任务？感觉应该快的话一周就能搞定，貌似不需要那么多创新，完成要求就行了。主要得快，前两名才能满分。

<img src="https://20220923img.oss-cn-hangzhou.aliyuncs.com/markdown/playground.png" width=700>

# PlayGround 技术报告

## 一、基础模块实现

### 1、Adam优化器的实现

nn.ts 中 updateWeightAdam 函数实现 Adam 的主干逻辑。

```typescript
export function updateWeightsAdam(network: Node[][], learningRate: number,batchSize:number,
                                  beta1: number = 0.99, beta2: number = 0.999, epsilon: number = 10e-8)
```

### 2、Normalization的实现

利用 typescript 的 Type 关键字构建抽象类 NormLayer

```typescript
type NormLayer = {
    alpha: number[];
    delta: number[];
    dispersion: number[];
    inputData: number[][];
    outputData: number[][];
    dAlpha: number[];
    dDelta: number[];
    dInput: number[][];

    m_t_alpha: number[];
    m_t_delta: number[];
    v_t_alpha: number[];
    v_t_delta: number[];

    forward(X: number[][],mode:string): number[][];
    backward(dOutput: number[][]): number[][];
};
```

继承 NormLayer 分别实现 LN 与 BN 的前传和反传，具体实现在 nn.ts 中的 LayerNormalization 类和 BatchNormalization 类。

## 二、基础模块嵌入

### 1、Adam的嵌入

playground.ts 中，通过判断 state 记录的 optimizer 参数选择更新方式

```typescript
if (state.optimizer === 0) {
    nn.updateWeightsSGD(network,
        state.learningRate,
        state.regularizationRate,
        state.batchSize);
} else if (state.optimizer === 1) {
    nn.updateWeightsAdam(network,
        state.learningRate,
        state.batchSize)
}
```

### 2、Normalization的嵌入

首先在 playground.ts 中的 reset 函数中，根据已有的参数初始化 normLayerList（保存了所有FC对应的 LN 或 BN 信息）

```typescript
if(state.normalization == 0) {
        normLayerList = null
    } else {
        normLayerList = {}
        for (let i = 1; i < network.length-2; i++) {
            normLayerList[i] = {}
            if(state.normalization == 1) {
                normLayerList[i]['layer'] = new BatchNormalization(shape[i]);
                normLayerList[i]['place'] = state.normPlace; 
            } else if(state.normalization == 2) {
                normLayerList[i]['layer'] = new LayerNormalization(shape[i])
                normLayerList[i]['place'] = state.normPlace;
            }
        }
    }
```

在 nn.ts 将面向点更新的部分，封装为面向层更新，包括前传和后传部分

```typescript
class LayerMethod {
    public static layerInput(layer: Node[], batchSize: number);
    public static layerActivate(layer: Node[], batchSize: number);
    public static layerActivateDir(layer: Node[], batchSize: number, NormResult: number[][]);
    public static layerDInput(layer: Node[], batchSize: number);
    public static layerDInputDir(layer: Node[], batchSize: number, normOutputDer: number[][]);
    public static layerDLink(layer: Node[], batchSize: number);
    public static layerDLinkDir(layer: Node[], batchSize: number, normInputDer: number[][]);
    public static layerDOutput(prevLayer: Node[], batchSize: number);
    public static constructNormInput(layer: Node[], type: number): number[][];
}
```

最后利用面向层更新方法，重构前传和反传的逻辑，具体见 nn.ts 中的 forwardProp 和 backProp 函数。

## 三、前端呈现

### 1、收敛迭代次数记录

![image-20230512210207432](/img/loss.png)

```typescript
    if (iter == 1) {
        state.initial_loss = lossTrain
    } else {
        if (state.converge_epoch == 0 && lossTrain <= 0.10 * state.initial_loss) {
            state.converge_epoch = iter
        }
    }
```

判断收敛条件为收敛到最初 loss 的 10%，便于分析前后参数改变，迭代次数的影响。

### 2、热力图展示 normalization 的结果

![image-20230512210239339](/img/feature.png)

```typescript
if (state.normalization != normalizations['none']) {
            nodeGroup.append("rect")
                .attr({
                    id: `norm-${nodeId}`,
                    x: RECT_SIZE + NORM_SIZE,
                    y: NORM_SIZE - NORM_SIZE,
                    width: 2 * NORM_SIZE,
                    height: 2 * NORM_SIZE,
                })
            // 插入 canvas
            let div = d3.select("#network").insert("div", ":first-child")
                .attr({
                    "id": `canvas-norm-${nodeId}`,
                    "class": "canvas"
                })
                .style({
                    position: "absolute",
                    left: `${x + RECT_SIZE + NORM_SIZE + 3}px`,
                    top: `${y}px`
                }).on("mouseenter", function () {
                    updateDecisionBoundary(network, false);
                    easyHover(1, "result after normalization", [x + RECT_SIZE + NORM_SIZE + 3, y]);
                    heatMap.updateBackground(boundary[nodeId]['norm'], state.discretize);
                }).on("mouseleave", function () {
                    updateDecisionBoundary(network, false);
                    easyHover(null);
                    heatMap.updateBackground(boundary[nn.getOutputNode(network).id]['non_norm'],
                        state.discretize);
                })
            let nodeHeatMap = new HeatMap(2 * NORM_SIZE, DENSITY / 10, xDomain,
                xDomain, div, {noSvg: true});
            div.datum({heatmap: nodeHeatMap, id: nodeId, type: 1});
        }
```

在playground.ts 的 drawNode 函数中依据现有参数绘制新增热力图。热力图的显示是根据 playground 中的 boundary 变量绘制的，boundary 中记录了每个节点对应的 heatmap 数据分布，我们在 boundary 中同时记录了每个节点有无 normalization 的结果，绘制时分别依据这两个结果得到两个 heatmap。


/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
/**
 * A node in a neural network. Each node has a state
 * (total input, output, and their respectively derivatives) which changes
 * after every forward and back propagation run.
 */
export class Node {
    id: string;
    /** List of input links. */
    inputLinks: Link[] = [];
    bias = 0.1;
    /** List of output links. */
    outputs: Link[] = [];
    totalInput: number[] = [];
    output: number[] = [];
    averageOutput: number;
    averageOutputNotNorm: number;
    /** Error derivative with respect to this node's output. */
    outputDer: number[] = [];
    /** Error derivative with respect to this node's total input. */
    inputDer: number[] = [];
    /** Activation function that takes total input and returns node's output */
    activation: ActivationFunction;
    /** 1:batchnorm 2:layernorm */
    normalization: number

    /**normalization layer */
    normlayer: NormLayer

    /**
     * Creates a new node with the provided id and activation function.
     */
    constructor(id: string, normalization: number, activation: ActivationFunction, initZero?: boolean) {
        this.id = id;
        this.normalization = normalization
        this.activation = activation;
        if (initZero) {
            this.bias = 0;
        }
    }

    /** Recomputes the node's output and returns it. */
    public updateOutput(batchSize: number): number[] {
        for (let i = 0; i < batchSize; i++) {
            this.totalInput[i] = this.bias;
        }
        // sum the link: weight * source
        for (let j = 0; j < this.inputLinks.length; j++) {
            let link = this.inputLinks[j];
            if(isNaN(link.weight)) {
                console.log(link);
            }
            for (let i = 0; i < batchSize; i++) {
                this.totalInput[i] += link.weight * link.source.output[i];
            }
        }
        this.averageOutput = 0
        for (let i = 0; i < batchSize; i++) {
            this.output[i] = this.activation.output(this.totalInput[i]);
            this.averageOutput += this.output[i]
        }
        this.averageOutput /= this.output.length
        this.averageOutputNotNorm = this.averageOutput;
        return this.output;
    }
}

class LayerMethod {
    public static layerInput(layer: Node[], batchSize: number) {
        /** to update node's total input at layer level */
        for (let i = 0; i < layer.length; i++) {
            let node = layer[i]
            for (let j = 0; j < batchSize; j++) {
                node.totalInput[j] = node.bias;
            }
            // sum the link: weight * source
            for (let j = 0; j < node.inputLinks.length; j++) {
                let link = node.inputLinks[j];
                if(isNaN(link.weight)) {
                    console.log(node.id + link)
                }
                for (let k = 0; k < batchSize; k++) {
                    if(isNaN(node.totalInput[k])) {
                        console.log("totalInput is nan: "+node.id)
                    }
                    node.totalInput[k] += link.weight * link.source.output[k];
                }
            }
            // console.log("batchSize = "+batchSize+", totalInput: "+node.totalInput)
        }
    };

    public static layerActivate(layer: Node[], batchSize: number) {
        /** to compute the activation at layer level*/
        for (let i = 0; i < layer.length; i++) {
            let node = layer[i]
            node.averageOutput = 0
            for (let j = 0; j < batchSize; j++) {
                node.output[j] = node.activation.output(node.totalInput[j]);
                if(isNaN(node.output[j])) {
                    console.log("node output is nan: "+node.id);
                }
                node.averageOutput += node.output[j]
            }
            node.averageOutput /= node.output.length
            node.averageOutputNotNorm = node.averageOutput;
        }
    };

    public static layerDInput(layer: Node[], batchSize: number) {
        for (let i = 0; i < layer.length; i++) {
            let node = layer[i];
            for (let j = 0; j < batchSize; j++) {
                node.inputDer[j] = node.outputDer[j] * node.activation.der(node.totalInput[j]);
            }
        }
    }

    public static layerDLink(layer: Node[], batchSize: number) {
        // Error derivative with respect to each weight coming into the node.
        for (let i = 0; i < layer.length; i++) {
            let node = layer[i];
            for (let j = 0; j < node.inputLinks.length; j++) {
                let link = node.inputLinks[j];
                if (link.isDead) {
                    continue;
                }
                for (let k = 0; k < batchSize; k++) {
                    link.errorDer = node.inputDer[k] * link.source.output[k];
                    link.accErrorDer += link.errorDer;
                    link.numAccumulatedDers++;
                }
            }
        }
    }

    public static layerDOutput(prevLayer: Node[], batchSize: number) {
        for (let i = 0; i < prevLayer.length; i++) {
            let node = prevLayer[i];
            // Compute the error derivative with respect to each node's output.
            for (let j = 0; j < batchSize; j++) {
                node.outputDer[j] = 0;
            }
            for (let j = 0; j < node.outputs.length; j++) {
                let output = node.outputs[j];
                for (let k = 0; k < batchSize; k++) {
                    node.outputDer[k] += output.weight * output.dest.inputDer[k];
                }
            }
        }
    }


    public static constructNormInput(layer: Node[], type: number): number[][] {
        /**
         * type = 0: 获取activation前的结果
         * type = 1: 获取activation后的结果
         * */
        let input = []  // 构造的input，其每列为单条数据
        for (let i = 0; i < layer.length; i++) {
            let node = layer[i];
            if (type == 0) {
                // console.log("node.totalInput: "+node.totalInput)
                input[i] = Copy1D(node.totalInput);
            } else if (type == 1) {
                // console.log("node.output: "+node.output)
                input[i] = Copy1D(node.output);
            } else if(type == 2) {
                input[i] = Copy1D(node.inputDer);
            } else if(type == 3) {
                input[i] = Copy1D(node.outputDer);
            }
        }
        return input;
    }

    public static setNormOutput(layer: Node[], normOutput: number[][], type: number) {
        for (let i = 0; i < layer.length; i++) {
            let node = layer[i];
            if (type == 0) {
                node.totalInput = Copy1D(normOutput[i]);
            } else if (type == 1) {
                node.output = Copy1D(normOutput[i]);
            } else if(type == 2) {
                node.inputDer = Copy1D(normOutput[i]);
            } else if(type == 3) {
                node.outputDer = Copy1D(normOutput[i]);
            }
        }
    }
}

function Copy(X: number[][]): number[][] {
    let copy = [];
    for (let i = 0; i < X.length; i++) {
        copy[i] = [];
        for (let j = 0; j < X[i].length; j++) {
            copy[i][j] = X[i][j];
        }
    }
    return copy;
}

function Copy1D(X: number[]): number[] {
    let copy = [];
    for (let i = 0; i < X.length; i++) {
        copy[i] = X[i];
    }
    return copy;
}

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

    forward(X: number[][]): number[][];
    backward(dOutput: number[][]): number[][];
};

export class BatchNormalization implements NormLayer {
    dispersion: number[];
    avgMoving: number[];
    varMoving: number[];
    batchAvg: number[];
    batchVar: number[];
    alpha: number[];
    delta: number[];
    eps: number;
    rateDecay: number;
    Xnorm: number[][];
    inputData: number[][];
    outputData: number[][];

    m_t_alpha: number[];
    m_t_delta: number[];
    v_t_alpha: number[];
    v_t_delta: number[];

    dAlpha: number[];
    dDelta: number[];
    dInput: number[][];

    constructor(dim: number) {
        this.avgMoving = new Array(dim);
        this.varMoving = new Array(dim);
        this.alpha = new Array(dim);
        this.delta = new Array(dim);
        this.batchAvg = new Array(dim);
        this.batchVar = new Array(dim);
        this.eps = 1e-5;
        this.rateDecay = 0.95;
        this.m_t_alpha = new Array(dim);
        this.m_t_delta = new Array(dim);
        this.v_t_alpha = new Array(dim);
        this.v_t_delta = new Array(dim);

        for (let i = 0; i < dim; i++) {
            this.avgMoving[i] = 0;
            this.varMoving[i] = 0;
            this.alpha[i] = 1;
            this.delta[i] = 0;
            this.batchAvg[i] = 0;
            this.batchVar[i] = 0;
            this.m_t_alpha[i] = 0;
            this.m_t_delta[i] = 0;
            this.v_t_alpha[i] = 0;
            this.v_t_delta[i] = 0;
        }
    }

    forward(X: number[][],mode:string): number[][] {
        // console.log('here!');
        this.inputData = Copy(X);
        this.outputData = Copy(X);
        let Xnorm = Copy(X);
        let L = X.length;
        let N = X[0].length;
        if (N == 1) {
            return this.inputData;
        }
        let average = new Array(L);
        let dispersion = new Array(L);
        //如果不是训练模式,固定的在训练中得出的mean和var计算：
        if (mode === 'eval') {
            for (let i = 0; i < L; i++) {
                for (let j = 0; j < N; j++) {
                    Xnorm[i][j] = (X[i][j] - this.avgMoving[i]) / Math.sqrt(this.varMoving[i] + this.eps);
                    this.outputData[i][j] = this.alpha[i] * Xnorm[i][j] + this.delta[i];
                }
            }
            this.Xnorm = Copy(Xnorm);
            return this.outputData;
        }

        for (let i = 0; i < L; i++) {
            average[i] = 0;
            dispersion[i] = 0;
            for (let j = 0; j < N; j++) {
                average[i] += X[i][j];
            }
            average[i] /= N;
            for (let j = 0; j < N; j++) {
                dispersion[i] += (X[i][j] - average[i]) ^ 2;
            }
            dispersion[i] = dispersion[i] / N;
            for (let j = 0; j < N; j++) {
                Xnorm[i][j] = (X[i][j] - average[i]) / Math.sqrt(dispersion[i] + this.eps);
                this.outputData[i][j] = this.alpha[i] * Xnorm[i][j] + this.delta[i];
            }
            this.avgMoving[i] = this.rateDecay * this.avgMoving[i] + (1 - this.rateDecay) * average[i];
            this.varMoving[i] = this.rateDecay * this.varMoving[i] + (1 - this.rateDecay) * dispersion[i];
        }
        this.Xnorm = Copy(Xnorm);
        this.batchAvg = Copy1D(average);
        this.batchVar = Copy1D(dispersion);
        return this.outputData;
    }

    backward(dOutput: number[][]): number[][] {
        let L = dOutput.length;
        let N = dOutput[0].length;
        this.dInput = Copy(dOutput);
        if (N == 1) {
            return this.dInput;
        }
        this.dAlpha = new Array(N);
        this.dDelta = new Array(N);
        for (let i = 0; i < L; i++) {
            this.dDelta[i] = 0;
            this.dAlpha[i] = 0;
            for (let j = 0; j < N; j++) {
                this.dDelta[i] += dOutput[i][j];
                this.dAlpha[i] += dOutput[i][j] * this.Xnorm[i][j];
            }
        }
        for (let i = 0; i < L; i++) {
            let dDisp = 0;
            let dAvg = 0;
            let k = Math.sqrt((this.batchVar[i] + this.eps) * (this.batchVar[i] + this.eps) * (this.batchVar[i] + this.eps));
            for (let j = 0; j < N; j++) {
                dDisp += dOutput[i][j] * this.alpha[i] * (this.inputData[i][j] - this.batchAvg[i]) * -0.5 / k;
                dAvg += -dOutput[i][j] * this.alpha[i] / Math.sqrt(this.batchVar[i] + this.eps);
            }
            for (let j = 0; j < N; j++) {
                this.dInput[i][j] += dOutput[i][j] * this.alpha[i] / Math.sqrt(this.batchVar[i] + this.eps) + dDisp * 2 * (this.inputData[i][j] - this.batchAvg[i]) / N + dAvg / N;
            }
        }
        return this.dInput;
    }
}

export class LayerNormalization implements NormLayer {

    alpha: number[];
    delta: number[];
    epsVal: number;
    inputData: number[][];
    Xnorm: number[][];
    outputData: number[][];
    varData: number[];
    dAlpha: number[];
    dDelta: number[];
    dInput: number[][];
    m_t_alpha: number[];
    m_t_delta: number[];
    v_t_alpha: number[];
    v_t_delta: number[];

    constructor(dim: number) {
        this.alpha = new Array(dim);
        this.delta = new Array(dim);
        this.m_t_alpha = new Array(dim);
        this.m_t_delta = new Array(dim);
        this.v_t_alpha = new Array(dim);
        this.v_t_delta = new Array(dim);
        this.epsVal = 1e-5;
        for (let i = 0; i < dim; i++) {
            this.alpha[i] = 1;
            this.delta[i] = 0;
            this.m_t_alpha[i] = 0;
            this.m_t_delta[i] = 0;
            this.v_t_alpha[i] = 0;
            this.v_t_delta[i] = 0;
        }
    }
    forward(X: number[][]): number[][] {
        this.inputData = Copy(X);
        this.outputData = Copy(X);
        let D = X.length;
        let N = X[0].length;
        if(N == 1) {
            return this.outputData;
        }
        let avg = new Array(N);
        let varData = new Array(N);
        let Xnorm = Copy(X);

        for (let i = 0; i < N; i++) {
            avg[i] = 0;
            varData[i] = 0;
            for (let j = 0; j < D; j++) {
                avg[i] += X[j][i];
            }
            avg[i] /= D;
            for (let j = 0; j < D; j++) {
                varData[i] += (X[j][i] - avg[i]) * (X[j][i] - avg[i]);
            }
            varData[i] = Math.sqrt(varData[i] / D + this.epsVal);
            for (let j = 0; j < D; j++) {
                Xnorm[j][i] = (X[j][i] - avg[i]) / varData[i];
                this.outputData[j][i] = this.alpha[j] * Xnorm[j][i] + this.delta[j];
            }
        }
        this.Xnorm = Copy(Xnorm);
        this.varData = Copy1D(varData);
        return this.outputData;
    }

    backward(dOutput: number[][]): number[][] {
        let D = dOutput.length;
        let N = dOutput[0].length;
        this.dInput = Copy(dOutput);
        if(N == 1) {
            return this.dInput;
        }
        this.dAlpha = new Array(D);
        this.dDelta = new Array(D);
        for (let j = 0; j < D; j++) {
            this.dDelta[j] = 0;
            this.dAlpha[j] = 0;
            for (let i = 0; i < N; i++) {
                this.dDelta[j] += dOutput[j][i];
                this.dAlpha[j] += dOutput[j][i] * this.Xnorm[j][i];
            }
        }
        for (let i = 0; i < N; i++) {
            let sumdXnorm = 0;
            let sumdXnormX = 0;
            for (let k = 0; k < D; k++) {
                sumdXnorm += (dOutput[k][i] * this.alpha[k]);
                sumdXnormX += (dOutput[k][i] * this.alpha[k] * this.Xnorm[k][i]);
            }
            for (let j = 0; j < D; j++) {
                this.dInput[j][i] = 1 / (D * this.varData[i]) * (D * dOutput[j][i] * this.alpha[j] - this.Xnorm[j][i] * sumdXnormX - sumdXnorm);
            }
        }
        return this.dInput;
    }

    dispersion: number[];

}


/**
 * An error function and its derivative.
 */
export interface ErrorFunction {
    error: (output: number, target: number) => number;
    der: (output: number, target: number) => number;
}

/** A node's activation function and its derivative. */
export interface ActivationFunction {
    output: (input: number) => number;
    der: (input: number) => number;
}

/** Function that computes a penalty cost for a given weight in the network. */
export interface RegularizationFunction {
    output: (weight: number) => number;
    der: (weight: number) => number;
}

/** Built-in error functions */
export class Errors {
    public static SQUARE: ErrorFunction = {
        error: (output: number, target: number) =>
            0.5 * Math.pow(output - target, 2),
        der: (output: number, target: number) => output - target
    };
}

/** Polyfill for TANH */
(Math as any).tanh = (Math as any).tanh || function (x) {
    if (x === Infinity) {
        return 1;
    } else if (x === -Infinity) {
        return -1;
    } else {
        let e2x = Math.exp(2 * x);
        return (e2x - 1) / (e2x + 1);
    }
};

/** Built-in activation functions */
export class Activations {
    public static TANH: ActivationFunction = {
        output: x => (Math as any).tanh(x),
        der: x => {
            let output = Activations.TANH.output(x);
            return 1 - output * output;
        }
    };
    public static RELU: ActivationFunction = {
        output: x => Math.max(0, x),
        der: x => x <= 0 ? 0 : 1
    };
    public static SIGMOID: ActivationFunction = {
        output: x => 1 / (1 + Math.exp(-x)),
        der: x => {
            let output = Activations.SIGMOID.output(x);
            return output * (1 - output);
        }
    };
    public static LINEAR: ActivationFunction = {
        output: x => x,
        der: x => 1
    };
}


/** Build-in regularization functions */
export class RegularizationFunction {
    public static L1: RegularizationFunction = {
        output: w => Math.abs(w),
        der: w => w < 0 ? -1 : (w > 0 ? 1 : 0)
    };
    public static L2: RegularizationFunction = {
        output: w => 0.5 * w * w,
        der: w => w
    };
}

/**
 * A link in a neural network. Each link has a weight and a source and
 * destination node. Also it has an internal state (error derivative
 * with respect to a particular input) which gets updated after
 * a run of back propagation.
 */
export class Link {
    id: string;
    source: Node;
    dest: Node;
    weight = Math.random() - 0.5;
    isDead = false;
    /** Error derivative with respect to this weight. */
    errorDer = 0;
    /** Accumulated error derivative since the last update. */
    accErrorDer = 0;
    /** Number of accumulated derivatives since the last update. */
    numAccumulatedDers = 0;
    regularization: RegularizationFunction;

    /**
     * Constructs a link in the neural network initialized with random weight.
     *
     * @param source The source node.
     * @param dest The destination node.
     * @param regularization The regularization function that computes the
     *     penalty for this weight. If null, there will be no regularization.
     */
    constructor(source: Node, dest: Node,
                regularization: RegularizationFunction, initZero?: boolean) {
        this.id = source.id + "-" + dest.id;
        this.source = source;
        this.dest = dest;
        this.regularization = regularization;
        if (initZero) {
            this.weight = 0;
        }
    }
}


/**
 * Builds a neural network.
 *
 * @param networkShape The shape of the network. E.g. [1, 2, 3, 1] means
 *   the network will have one input node, 2 nodes in first hidden layer,
 *   3 nodes in second hidden layer and 1 output node.
 * @param activation The activation function of every hidden node.
 * @param outputActivation The activation function for the output nodes.
 * @param regularization The regularization function that computes a penalty
 *     for a given weight (parameter) in the network. If null, there will be
 *     no regularization.
 * @param inputIds List of ids for the input nodes.
 */

// Add `useBatchNormalization` and `useLayerNormalization` to the function parameters.
export function buildNetwork(
    networkShape: number[], normalization: number, activation: ActivationFunction,
    outputActivation: ActivationFunction,
    regularization: RegularizationFunction,
    inputIds: string[], initZero?: boolean,
): Node[][] {
    let numLayers = networkShape.length;
    let id = 1;
    /** List of layers, with each layer being a list of nodes. */
    let network: Node[][] = [];
    for (let layerIdx = 0; layerIdx < numLayers; layerIdx++) {
        let isOutputLayer = layerIdx === numLayers - 1;
        let isInputLayer = layerIdx === 0;
        let currentLayer: Node[] = [];
        network.push(currentLayer);
        let numNodes = networkShape[layerIdx];
        let normlayer: NormLayer;
        // Add Batch Normalization or Layer Normalization if requested.
        if (normalization === 1 && !isInputLayer && !isOutputLayer) {
            normlayer = new BatchNormalization(numNodes);
        }
        if (normalization === 2 && !isInputLayer && !isOutputLayer) {
            normlayer = new LayerNormalization(numNodes);
        }
        for (let i = 0; i < numNodes; i++) {
            let nodeId = id.toString();
            if (isInputLayer) {
                nodeId = inputIds[i];
            } else {
                id++;
            }
            let node = new Node(nodeId, normalization,
                isOutputLayer ? outputActivation : activation, initZero);
            if (normalization > 0 && ~isInputLayer && ~isOutputLayer) node.normlayer = normlayer;
            currentLayer.push(node);
            if (layerIdx >= 1) {
                // Add links from nodes in the previous layer to this node.
                for (let j = 0; j < network[layerIdx - 1].length; j++) {
                    let prevNode = network[layerIdx - 1][j];
                    let link = new Link(prevNode, node, regularization, initZero);
                    prevNode.outputs.push(link);
                    node.inputLinks.push(link);
                }
            }
        }
    }
    return network;
}


/**
 * Runs a forward propagation of the provided input through the provided
 * network. This method modifies the internal state of the network - the
 * total input and output of each node in the network.
 *
 * @param network The neural network.
 * @param inputs The input array. Its length should match the number of input
 *     nodes in the network.
 * @param batchSize
 * @param normLayerList
 * @return The final output of the network.
 */
export function forwardProp(network: Node[][], inputs: number[][], batchSize: number,
                            normLayerList: { [layerNum: number]: { layer: NormLayer, place: number } }): number[] {
    let inputLayer = network[0];
    if (inputs[0].length !== inputLayer.length) {
        throw new Error("The number of inputs must match the number of nodes in" +
            " the input layer");
    }
    // Update the input layer.
    for (let i = 0; i < inputLayer.length; i++) {
        let node = inputLayer[i];
        for (let j = 0; j < batchSize; j++) {
            node.output[j] = inputs[j][i];  // 第 j 条数据的第 i 个特征
        }
    }
    for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
        /**
         * 如果没有 norm 这一步直接完成计算；
         * 如果有 norm 这一步完成 outputNotNorm 的计算
         * */
        let currentLayer = network[layerIdx];
        // Update all the nodes in this layer.
        LayerMethod.layerInput(currentLayer, batchSize);
        LayerMethod.layerActivate(currentLayer, batchSize);
    }
    if (normLayerList != null) {
        for (let layerIdx = 1; layerIdx < network.length - 1; layerIdx++) {
            let currentLayer = network[layerIdx];
            if (layerIdx in normLayerList) {
                let place = normLayerList[layerIdx]['place'];
                let normLayer = normLayerList[layerIdx]['layer'];
<<<<<<< Updated upstream
                let input = NodeLayerMethod.constructNormInput(currentLayer, place);
                 //console.log(input)
                let normResult = normLayer.forward(input);
                //console.log(normResult)
                NodeLayerMethod.setNormOutput(currentLayer, normResult, place);
                NodeLayerMethod.layerActivate(currentLayer, batchSize);
=======
                let input = LayerMethod.constructNormInput(currentLayer, place);
                // console.log(input)   // checkpoint
                let normResult = normLayer.forward(input);
                // console.log(normResult)
                LayerMethod.setNormOutput(currentLayer, normResult, place);
                LayerMethod.layerActivate(currentLayer, batchSize);
>>>>>>> Stashed changes
            } else {
                LayerMethod.layerInput(currentLayer, batchSize);
                LayerMethod.layerActivate(currentLayer, batchSize);
            }
        }
        /** 最后一层的 output node 只需更新输出 */
        let currentLayer = network[network.length - 1];
        currentLayer[0].updateOutput(batchSize);
    }

    return network[network.length - 1][0].output;
}

/**
 * Runs a backward propagation using the provided target and the
 * computed output of the previous call to forward propagation.
 * This method modifies the internal state of the network - the error
 * derivatives with respect to each node, and each weight
 * in the network.
 */
export function backProp(network: Node[][], target: number[],
                         errorFunc: ErrorFunction, batchSize: number,
                         normLayerList: { [layerNum: number]: { layer: NormLayer, place: number } }): void {
    // The output node is a special case. We use the user-defined error
    // function for the derivative.
    let outputNode = network[network.length - 1][0];
    for (let i = 0; i < batchSize; i++) {
        outputNode.outputDer[i] = errorFunc.der(outputNode.output[i], target[i]);
    }
    // Go through the layers backwards.
    for (let layerIdx = network.length - 1; layerIdx >= 1; layerIdx--) {
        let currentLayer = network[layerIdx];
        // Compute the error derivative of each node with respect to:
        // 1) its total input
        // 2) each of its input weights.
        LayerMethod.layerDInput(currentLayer, batchSize);
        // Error derivative with respect to each weight coming into the node.
        // if(normLayerList != null && layerIdx in normLayerList) {
        //     let place = normLayerList[layerIdx]['place'];
        //     let normLayer = normLayerList[layerIdx]['layer'];
        //     let input = LayerMethod.constructNormInput(currentLayer, place+2);
        //     // console.log(input)   // checkpoint
        //     let normResult = normLayer.backward(input);
        //     LayerMethod.setNormOutput(currentLayer, normResult, place+2);
        //     if(place == 1) {
        //         LayerMethod.layerDInput(currentLayer, batchSize);
        //     }
        // }
        LayerMethod.layerDLink(currentLayer, batchSize);
        // Compute derivative with respect to prev layer's output
        if (layerIdx === 1) {
            continue;
        }
        let prevLayer = network[layerIdx - 1];
        LayerMethod.layerDOutput(prevLayer, batchSize);
    }
}

/**
 * Updates the weights of the network using the previously accumulated error
 * derivatives.
 */
/**
 * 使用Adam方法更新神经网络中的权重和偏置。
 * @param network - 神经网络，由节点层组成的二维数组。
 * @param learningRate - 学习率，表示在每次迭代中应用于权重和偏置的更新量。
 * @param beta1 - Adam方法的一阶动量衰减率。
 * @param beta2 - Adam方法的二阶动量衰减率。
 * @param epsilon - 防止除零错误的小量值。
 */
export function updateWeightsAdam(network: Node[][], learningRate: number,
                                  beta1: number = 0.99, beta2: number = 0.999, epsilon: number = 10e-8) {
    let m: number[][][] = [];  // 一阶动量
    let v: number[][][] = [];  // 二阶动量
    let t = 0;  // 时间步长
    for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
        let currentLayer = network[layerIdx];
        m[layerIdx] = new Array(currentLayer.length);
        v[layerIdx] = new Array(currentLayer.length);
        for (let i = 0; i < currentLayer.length; i++) {
            let node = currentLayer[i];
            // 初始化一阶动量和二阶动量
            m[layerIdx][i] = new Array(node.inputLinks.length + 1);
            v[layerIdx][i] = new Array(node.inputLinks.length + 1);
            // 更新节点的偏置
            let numAccumulatedDers = node.inputDer.length
            let accInputDer = 0
            for (let j = 0; j < node.inputDer.length; j++) {
                accInputDer += node.inputDer[j];
            }
            if (numAccumulatedDers > 0) {
                node.bias -= learningRate * accInputDer / numAccumulatedDers;
                for (let j = 0; j < node.inputDer.length; j++) {
                    node.inputDer[i] = 0;
                }
            }
            // 更新输入到该节点的权重
            for (let j = 0; j < node.inputLinks.length; j++) {
                let link = node.inputLinks[j];
                if (link.isDead) {
                    continue;
                }
                if (m[layerIdx][i][j] === undefined) {
                    m[layerIdx][i][j] = 0;
                }
                if (v[layerIdx][i][j] === undefined) {
                    v[layerIdx][i][j] = 0;
                }
                if (link.numAccumulatedDers > 0) {
                    // 更新一阶动量和二阶动量
                    m[layerIdx][i][j] = beta1 * m[layerIdx][i][j] + (1 - beta1) * link.accErrorDer / link.numAccumulatedDers;
                    v[layerIdx][i][j] = beta2 * v[layerIdx][i][j] + (1 - beta2) * Math.pow(link.accErrorDer / link.numAccumulatedDers, 2);
                    t += 1;
                    // 计算修正后的一阶动量和二阶动量
                    let mHat = m[layerIdx][i][j] / (1 - Math.pow(beta1, t));
                    let vHat = v[layerIdx][i][j] / (1 - Math.pow(beta2, t));
                    // 更新权重
                    link.weight -= (learningRate / (Math.sqrt(vHat) + epsilon)) * mHat;
                    // 重置累计梯度
                    link.accErrorDer = 0;
                    link.numAccumulatedDers = 0;
                }
            }
        }
    }
}

export function updateWeightsSGD(network: Node[][], learningRate: number,
                                 regularizationRate: number) {
    for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
        let currentLayer = network[layerIdx];
        for (let i = 0; i < currentLayer.length; i++) {
            let node = currentLayer[i];
            // Update the node's bias.
            let numAccumulatedDers = node.inputDer.length
            let accInputDer = 0
            for (let j = 0; j < node.inputDer.length; j++) {
                accInputDer += node.inputDer[j];
            }
            if (numAccumulatedDers > 0) {
                node.bias -= learningRate * accInputDer / numAccumulatedDers;
                for (let j = 0; j < node.inputDer.length; j++) {
                    node.inputDer[i] = 0;
                }
            }
            // Update the weights coming into this node.
            for (let j = 0; j < node.inputLinks.length; j++) {
                let link = node.inputLinks[j];
                if (link.isDead) {
                    continue;
                }
                let regulDer = link.regularization ?
                    link.regularization.der(link.weight) : 0;
                if (link.numAccumulatedDers > 0) {
                    // Update the weight based on dE/dw.
                    if(link.numAccumulatedDers == 0) {
                        console.error("link numAccumulatedDers == 0");
                    }
                    link.weight -= (learningRate / link.numAccumulatedDers) * link.accErrorDer;
                    // Further update the weight based on regularization.
                    let newLinkWeight = link.weight -
                        (learningRate * regularizationRate) * regulDer;
                    if (link.regularization === RegularizationFunction.L1 &&
                        link.weight * newLinkWeight < 0) {
                        // The weight crossed 0 due to the regularization term. Set it to 0.
                        link.weight = 0;
                        link.isDead = true;
                    } else {
                        link.weight = newLinkWeight;
                    }
                    link.accErrorDer = 0;
                    link.numAccumulatedDers = 0;
                }
            }
        }
    }
}

/** Iterates over every node in the network */
export function forEachNode(network: Node[][], ignoreInputs: boolean,
                            accessor: (node: Node) => any) {
    for (let layerIdx = ignoreInputs ? 1 : 0;
         layerIdx < network.length;
         layerIdx++) {
        let currentLayer = network[layerIdx]; // 同样是个数组
        for (let i = 0; i < currentLayer.length; i++) {
            let node = currentLayer[i];
            accessor(node);
        }
    }
}

/** Returns the output node in the network. */
// 貌似不论是回归还是分类，最终都只有一个输出节点
export function getOutputNode(network: Node[][]) {
    return network[network.length - 1][0];
}

export const T = (ary: any[]) => {
    let ar = []
    for (let i = 0; i < ary[0].length; i++) {
        let cd = [];
        for (let j = 0; j < ary.length; j++) {
            cd.push(ary[j][i])
        }
        if (cd.length != 0) {
            ar.push(cd);
        }
    }
    return ar
}
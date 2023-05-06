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
  /** Error derivative with respect to this node's output. */
  outputDer: number[] = [];
  /** Error derivative with respect to this node's total input. */
  inputDer: number[] = [];
  /** Activation function that takes total input and returns node's output */
  activation: ActivationFunction;
  /** 1:batchnorm 2:layernorm */
  normalization: number

  /**normalization layer */
  normlayer: NormalizationLayer

  /**
   * Creates a new node with the provided id and activation function.
   */
  constructor(id: string,normalization:number, activation: ActivationFunction, initZero?: boolean) {
    this.id = id;
    this.normalization=normalization
    this.activation = activation;
    if (initZero) {
      this.bias = 0;
    }
  }

  /** Recomputes the node's output and returns it. */
  updateOutput(batchSize: number): number[] {
    for (let i = 0; i < batchSize; i++) {
      this.totalInput[i] = this.bias;
    }

    for (let j = 0; j < this.inputLinks.length; j++) {
      let link = this.inputLinks[j];
      for (let i = 0; i < batchSize; i++) {
        this.totalInput[i] += link.weight * link.source.output[i];
      }
    }
    this.averageOutput = 0
    // console.log("totalInput+weight: "+this.totalInput)
    for(let i = 0; i < batchSize; i++) {
      this.output[i] = this.activation.output(this.totalInput[i]);
      this.averageOutput += this.output[i]
    }
    this.averageOutput /= this.output.length
    return this.output;
  }
}


type NormLayer = {
  gamma: number[];
  beta: number[];
  variances: number[];
  input: number[][];
  output: number[][];
  dGamma: number[];
  dBeta: number[];
  dX: number[][];

  m_t_gamma: number[];
  m_t_beta: number[];
  v_t_gamma: number[];
  v_t_beta: number[];

  forward(X: number[][], mode: string): number[][];
  backward(dY: number[][]): number[][];
};

export class BatchNorm implements NormLayer {
  movingMean: number[];
  movingVar: number[];
  batchMean: number[];
  batchVar: number[];
  gamma: number[];
  beta: number[];
  epsilon: number;
  decayRate: number;
  Xhat: number[][];
  input: number[][];
  output: number[][];

  m_t_gamma: number[];
  m_t_beta: number[];
  v_t_gamma: number[];
  v_t_beta: number[];

  dGamma: number[];
  dBeta: number[];
  dX: number[][];

  constructor(width: number) {
    this.movingMean = new Array(width);
    this.movingVar = new Array(width);
    this.gamma = new Array(width);
    this.beta = new Array(width);
    this.epsilon = 1e-5;
    this.decayRate = 0.95;
    this.m_t_gamma = new Array(width);
    this.m_t_beta = new Array(width);
    this.v_t_gamma = new Array(width);
    this.v_t_beta = new Array(width);

    for (let i = 0; i < width; i++) {
      this.movingMean[i] = 0;
      this.movingVar[i] = 0;
      this.gamma[i] = 1;
      this.beta[i] = 0;
      this.m_t_gamma[i] = 0;
      this.m_t_beta[i] = 0;
      this.v_t_gamma[i] = 0;
      this.v_t_beta[i] = 0;
    }
  }

  forward(X: number[][], mode: string): number[][] {
    this.input = Copy(X);
    this.output = Copy(X);
    let Xhat = Copy(X);
    let L = X.length;
    let N = X[0].length;
    let mean = new Array(L);
    let variances = new Array(L);
    if (mode === 'eval') {
      for (let i = 0; i < L; i++) {
        for (let j = 0; j < N; j++) {
          Xhat[i][j] = (X[i][j] - this.movingMean[i]) / Math.sqrt(this.movingVar[i] + this.epsilon);
          this.output[i][j] = this.gamma[i] * Xhat[i][j] + this.beta[i];
        }
      }
      return this.output;
    }
    for (let i = 0; i < L; i++) {
      mean[i] = 0;
      variances[i] = 0;
      for (let j = 0; j < N; j++) {
        mean[i] += X[i][j];
      }
      mean[i] /= N;
      for (let j = 0; j < N; j++) {
        variances[i] += (X[i][j] - mean[i]) * (X[i][j] - mean[i]);
      }
      variances[i] = variances[i] / N;
      for (let j =0; j < N; j++) {
        Xhat[i][j] = (X[i][j] - mean[i]) / Math.sqrt(variances[i] + this.epsilon);
        this.output[i][j] = this.gamma[i] * Xhat[i][j] + this.beta[i];
      }
      this.movingMean[i] = this.decayRate * this.movingMean[i] + (1 - this.decayRate) * mean[i];
      this.movingVar[i] = this.decayRate * this.movingVar[i] + (1 - this.decayRate) * variances[i];
    }
    this.Xhat = Copy(Xhat);
    this.batchMean = Copy(mean);
    this.batchVar = Copy(variances);
    return this.output;
  }

  backward(dY: number[][]): number[][] {
    let L = dY.length;
    let N = dY[0].length;
    this.dX = Copy(dY);
    this.dGamma = new Array(N);
    this.dBeta = new Array(N);
    for (let i = 0; i < L; i++) {
      this.dBeta[i] = 0;
      this.dGamma[i] = 0;
      for (let j = 0; j < N; j++) {
        this.dBeta[i] += dY[i][j];
        this.dGamma[i] += dY[i][j] * this.Xhat[i][j];
      }
    }
    for (let i = 0; i < L; i++) {
      let dVar = 0;
      let dMean = 0;
      let k = Math.sqrt((this.batchVar[i] + this.epsilon) * (this.batchVar[i] + this.epsilon) * (this.batchVar[i] + this.epsilon));
      for (let j = 0; j < N; j++) {
        dVar += dY[i][j] * this.gamma[i] * (this.input[i][j] - this.batchMean[i]) * -0.5 / k;
        dMean += -dY[i][j] * this.gamma[i] / Math.sqrt(this.batchVar[i] + this.epsilon);
      }
      for (let j = 0; j < N; j++) {
        this.dX[i][j] += dY[i][j] * this.gamma[i] / Math.sqrt(this.batchVar[i] + this.epsilon) + dVar * 2 * (this.input[i][j] - this.batchMean[i]) / N + dMean / N;
      }
    }
    return this.dX;
  }
}


export class LayerNorm implements NormLayer {

  gamma: number[];
  beta: number[];
  epsilon: number;
  input: number[][];
  Xhat: number[][];
  output: number[][];
  variances: number[];
  dGamma: number[];
  dBeta: number[];
  dX: number[][];

  m_t_gamma: number[];
  m_t_beta: number[];
  v_t_gamma: number[];
  v_t_beta: number[];

  constructor(width: number) {
    this.gamma = new Array(width);
    this.beta = new Array(width);
    this.m_t_gamma = new Array(width);
    this.m_t_beta = new Array(width);
    this.v_t_gamma = new Array(width);
    this.v_t_beta = new Array(width);
    this.epsilon = 1e-5;
    for (let i = 0; i < width; i++) {
      this.gamma[i] = 1;
      this.beta[i] = 0;
      this.m_t_gamma[i] = 0;
      this.m_t_beta[i] = 0;
      this.v_t_gamma[i] = 0;
      this.v_t_beta[i] = 0;
    }
  }

  forward(X: number[][], mode: string): number[][] {

  }

  // layer normalization back propagation
  backward(dY: number[][]): number[][] {
   // todo: implement layer normalization back propagation
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
(Math as any).tanh = (Math as any).tanh || function(x) {
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
    networkShape: number[],normalization:number, activation: ActivationFunction,
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
    if (normalization === 1 && !isInputLayer && !isOutputLayer) {
      normlayer = new BatchNorm(numNodes);
    }
    if (normalization === 2 && !isInputLayer && !isOutputLayer) {
      normlayer = new LayerNorm(numNodes);
    }
    for (let i = 0; i < numNodes; i++) {
      let nodeId = id.toString();
      if (isInputLayer) {
        nodeId = inputIds[i];
      } else {
        id++;
      }
      let node = new Node(nodeId,
          isOutputLayer ? outputActivation : activation, initZero);
      currentLayer.push(node);
      if (layerIdx >= 1) {
        // Add links from nodes in the previous layer to this node.
        for (let j = 0; j < network[layerIdx - 1].length; j++) {
          let prevNode = network[layerIdx - 1][j];
          let link = new Link(prevNode, node, regularization, initZero);
          prevNode.outputs.push(link);
          node.inputLinks.push(link);
        }

        // Add Batch Normalization or Layer Normalization if requested.
        if (useBatchNormalization && !isInputLayer && !isOutputLayer) {
          let bn = new BatchNormalization();
          node.addNormalization(bn);
        } else if (useLayerNormalization && !isInputLayer && !isOutputLayer) {
          let ln = new LayerNormalization();
          node.addNormalization(ln);
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
 * @return The final output of the network.
 */
export function forwardProp(network: Node[][], inputs: number[][], batchSize: number): number[] {
  let inputLayer = network[0];
  if (inputs[0].length !== inputLayer.length) {
    throw new Error("The number of inputs must match the number of nodes in" +
        " the input layer");
  }
  // Update the input layer.
  for (let i = 0; i < inputLayer.length; i++) {
    let node = inputLayer[i];
    for(let j = 0; j < batchSize; j++) {
      node.output[j]=inputs[j][i];  // 第 j 条数据的第 i 个特征
    }
  }
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    // Update all the nodes in this layer.
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      node.updateOutput(batchSize);
    }
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
    errorFunc: ErrorFunction, batchSize: number): void {
  // The output node is a special case. We use the user-defined error
  // function for the derivative.
  let outputNode = network[network.length - 1][0];
  for(let i=0; i < batchSize; i++) {
    outputNode.outputDer[i] = errorFunc.der(outputNode.output[i], target[i]);
  }
  // Go through the layers backwards.
  for (let layerIdx = network.length - 1; layerIdx >= 1; layerIdx--) {
    let currentLayer = network[layerIdx];
    // Compute the error derivative of each node with respect to:
    // 1) its total input
    // 2) each of its input weights.
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      for(let j=0; j<batchSize; j++) {
        node.inputDer[j] = node.outputDer[j] * node.activation.der(node.totalInput[j]);
      }
    }
    // Error derivative with respect to each weight coming into the node.
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        if (link.isDead) {
          continue;
        }
        for(let k=0; k<batchSize; k++) {
          link.errorDer = node.inputDer[k] * link.source.output[k];
          link.accErrorDer += link.errorDer;
          link.numAccumulatedDers++;
        }
      }
    }
    // Compute derivative with respect to prev layer's output
    if (layerIdx === 1) {
      continue;
    }
    let prevLayer = network[layerIdx - 1];
    for (let i = 0; i < prevLayer.length; i++) {
      let node = prevLayer[i];
      // Compute the error derivative with respect to each node's output.
      for (let j = 0; j < batchSize; j++) {
        node.outputDer[j] = 0;
      }
      for (let j = 0; j < node.outputs.length; j++) {
        let output = node.outputs[j];
        for(let k=0; k < batchSize; k++) {
          node.outputDer[k] += output.weight * output.dest.inputDer[k];
        }
      }
    }
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
                                  beta1: number=0.99, beta2: number=0.999, epsilon: number=10e-8) {
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
      for(let j=0; j<node.inputDer.length; j++) {
        accInputDer += node.inputDer[j];
      }
      if (numAccumulatedDers > 0) {
        node.bias -= learningRate * accInputDer / numAccumulatedDers;
        for(let j=0; j<node.inputDer.length; j++) {
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
      for(let j=0; j<node.inputDer.length; j++) {
        accInputDer += node.inputDer[j];
      }
      if (numAccumulatedDers > 0) {
        node.bias -= learningRate * accInputDer / numAccumulatedDers;
        for(let j=0; j<node.inputDer.length; j++) {
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

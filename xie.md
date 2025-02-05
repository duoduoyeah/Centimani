data from off-chip memory to on-chip memory before commencing. In contrast, SambaNova SN30 minimizes off-chip memory access by caching intermediate results within on-chip memory. Graphcore Bow-IPU achieves data locality by employing a memory management technique that allocates training data across different memory hierarchies. Cerebras CS-2 adopts a unique strategy by maintaining all the parameters within on-chip memory at all times to facilitate high bandwidth access.

### 3.2 Observations and Modeling Trade-offs

Based on these analyses of typical AI accelerators, we have three observations to guide the design of performance model:

**Observation 1:** These AI accelerators differ significantly in hardware and software implementation, but have some commonalities in the memory subsystem and data management.

These distinguished hardware designs profoundly influence the substantial variations in the performance of these AI accelerators. These unique software designs are intricately linked to the underlying hardware configurations and play a pivotal role in the effectiveness of these accelerators. When attempting to adapt DNN training to a given accelerator, the amalgamation of these specialized designs introduces a higher level of performance uncertainty.

While AI accelerators come in various forms, they share similarities in how they organize memory and manage training data. Specifically, aspects such as utilizing on-chip memory and off-chip memory in hierarchical memory subsystem, along with data management methodologies, serve as unifying elements across various AI accelerator designs.

**Observation 2:** Predicting the execution time of the entire DNN training on AI accelerators is difficult, and hardware simulators offer limited assistance in this regard.

DNN model training is a complex process that involves multiple training stages, which involves a variety of DNN operators and requires different functions from AI accelerators. For example, the computation stage primarily involves compute and memory access units, while the communication stage is more related to network components. As a result, predicting the performance of the entire training process, which comprises distinct stages and operators, can lead to significant deviations.

Traditional hardware simulators, such as Gem5 [24] and ZSim [93], are highly configurable to evaluate different architectures. Real workloads can be executed under full-system mode to collect all details of the runs, including execution time. However, hardware simulators encounter two challenges, (1) The microarchitectures of these accelerators are not made public entirely. (2) Even with open-source hardware, simulators tend to significantly slow down programs [27], typically by a factor of 20x to 40x, making it infeasible to simulate the time-consuming DNN training tasks.

**Observation 3:** Choosing appropriate batch sizes must consider hardware limits and software optimizations.

The hardware limits of each AI accelerator can constrain the selection of batch size. For instance, SambaNova's on-chip memory is limited to 640 MB, which is designed to store all training data and intermediate results during DNN training, thus greatly limiting the maximum number of samples that can be processed simultaneously.

Based on these observations, we can identify three trade-offs that need to be considered in performance modeling. (1) If the execution time of DNN operators can be accurately predicted and appropriately combined, the overall prediction will be more accurate. (2) A uniform hardware abstraction that executes various DNN operators should participate in performance prediction, improving prediction accuracy, and reducing modeling complexity. (3) A uniform software abstraction, which manages training data and affects the selection of batch size under hardware limits, should be included.

## 4 Centimani: A Performance Predictor

### 4.1 Centimani Overview

Our performance predictor Centimani is inspired by three key trade-offs. To accommodate various hardware designs for software optimization, arising from the generality and effectiveness of performance modeling, two abstractions are presented: (1) **hardware abstraction** proposes a unified hardware interface that hides fine-grained characterizations of the underlying hardware design and reveals coarse-grained performance behaviors of different resources provided by AI accelerators; (2) **software abstraction** provides a unified mechanism that allows allocation, placement, and accessing of all training data on the hierarchical memory subsystem to meet various computation and transmission demands to enhance the manageability of DNN training on AI accelerators.

In addition, the two abstractions are connected through the **execution modeling,** which maps the DNN training process with associated hyperparameters into an efficient execution on specialized AI accelerators. This mapping is a critical step in revealing the real performance behaviors of each AI accelerator. The modeling process is shown in Figure 3. Firstly, the software abstraction provides control over hardware-related hyperparameters such as batch size (see Section 4.2); secondly, the execution model takes DNN model and optimal hyper-parameters to map three training stages into corresponding training components, and then each training component is further decomposed into a combination of various DNN operators; thirdly, the execution time of all the operators is predicted by decoupled performance models (see Section 4.3), and the overlap of these stages is removed to arrive at a final time prediction (see Section 4.3.4). To sum up, execution modeling provides a bridge between the high-level model description in deep learning frameworks and the low-level



![AI Accelerators Hardware and Software Abstractions](image.png)

Figure 3: Hardware and software abstractions of AI accelerators, and two corresponding performance modeling components (colored in navy blue) in Centimani.

---

hardware-specific instructions. In addition, the predicted execution time can be converted into different performance metrics, such as throughput or price-performance ratio, which can be used by end users (or researchers) to make informed hardware selections.

### 4.2 Hyper-parameters Selection

Selecting appropriate hyper-parameters is a crucial decision to effectively optimize training throughput and efficiency when training a DNN model on new AI accelerators. Some default but important hardware-related hyper-parameters can lead to inefficient hardware utilization [33, 74, 118], limited/exceeded parallelism [65, 77, 80], and resource contention [62]. We propose a novel memory estimation model to resolve the hesitancy of choosing the most important hyperparameter - batch size. We first show our preliminary results by comparing training throughputs across multiple batch sizes and then describe the proposed model.

#### 4.2.1 Large Difference across Multiple Batch Sizes

We implement two DNN models, e.g., ResNet-50 v1.5 [55] and Bert-Base [38], on four AI accelerators to study the difference in training throughput of using multiple batch sizes. The training throughput of these two models is collected respectively.

Table 2 summarizes the speedups of using optimal batch sizes compared to default batch sizes used on the GPU platform (NVIDIA A100 40GB) for four AI accelerators. It is obvious that it is unnecessary or even infeasible to tenaciously use default batch sizes on all AI accelerators. For example, training ResNet-50 on SambaNova SN30 or training Bert-Base on SambaNova SN30 and Cerebras CS-2 with default batch sizes will cause out-of-memory (OOM) errors. In addition, the default batch sizes cannot achieve the best training throughput, where the training throughput with optimal batch sizes of the two models can reach on average 4.68× and up to 9.31× higher training throughput than that with default batch sizes. In other words, the result further proved that choosing appropriate batch sizes is critical in using AI accelerators.

#### 4.2.2 Memory Model

Based on Observation 1 in Section 3.2, all AI accelerators employ a hierarchical memory subsystem and manage training data through their memory management mechanisms. The observation inspires our memory model to identify data placement across multilevel memory and differentiate behaviors in their properties. Unlike conventional CPU/GPU systems that put training data in the same memory hierarchy, e.g., placing all training data on DRAM or global memory for CPUs and GPUs, our memory model considers the configuration of multilevel memory and predicts the respective memory consumption of each memory level by introducing data classification and memory estimation model for a given batch size. From this, our memory model selects the optimal batch size that maximizes memory efficiency and avoids exceeding hardware limits using batch size selection.

**Data Classification** Data classification is designed on top of mainstream training frameworks, such as PyTorch, TensorFlow, and MXNet, which are supported by all AI accelerators and organize the execution of DNN training through a structural representation, known as a computation graph. In a computation graph, each node represents the invocation of a mathematical operator, such as matrix multiplication or concatenation, which takes tensor variables (multidimensional arrays) as input and output. Each mathematical operator incurs a certain memory overhead by math libraries (i.e., cuBLAS and cuDNN for NVIDIA GPUs). Each operator may contain numerical learnable parameters/tensors (i.e., weights and gradients). Additionally, execution dependencies are specified by edges that point from the output of one operator to the input of another. When a batch size for a DNN model is chosen, all tensor shapes involved in the computation graph are fixed. This characteristic is exploited by our memory model.

To fully understand how memory is consumed during DNN training, we classify the allocated data into five categories:

| AI Accelerators  | Models/ Dataset | Best Batch Size | Speedup Over Default | Default Setting |
|------------------|-----------------|---------------|----------------------|-----------------|
| SambaNova SN30   | ResNet-50       | 32              | Out-of-memory        | 256             |
| Graphcore Bow-IPU| ResNet-50       | 1024            | 4.17x                | 256             |
| Cerebras CS-2    | ResNet-50       | 480             | 2.77x                | 256             |
| Habana Gaudi2    | ResNet-50       | 512             | 1.94x                | 256             |
| SambaNova SN30   | Bert-Base       | 1004            | Out-of-memory        | 3              |
| Graphcore Bow-IPU| Bert-Base       | 16              | 9.31x                | 3              |
| Cerebras CS-2    | Bert-Base       | 2               | Out-of-memory        | 3              |
| Habana Gaudi2    | Squad2          | 12              | 5.24x                | 3              |


- Input/Output Tensors, which include input tensors and output tensors (such as activations). Activations are further computed to forward output and output gradients.
- W&B Tensors, which include weights and biases of operations. They are learnable parameters of training.
- Gradient Tensors, which include gradients, weight gradients, and gradients for momentum. They are computed under backward propagation for updating and calculating weights in the next iteration.
- Algorithm-related Tensors, which include variables used for specific algorithms such as mixed precision training [78] and stabilized SGD [61].
- Ephemeral Tensors, which include temporary variables used in operation implementation such as mathematical library and communication reservation used for multi-device training.

Altogether, such categorization offers a structured classification for comprehending the multifaceted roles and contributions of tensors in the memory consumption of DNN training.
Memory Estimation Model To build the memory estimation model, it is essential to consider both the memory consumption of various training data and their mapping to hierarchical memory system. The workflow of the memory estimation model can be divided into two parts as follows.

The first part is to traverse the computation graph of DNN models and infer the memory consumption of each operator and data in a fine-grained manner. For example, given a model with layers, we traverse each model layer in turn and infer the memory consumption of each operator in each model layer. For each model layer (e.g., layer L), the size of the activations of the preview layers (layer L − 1) and the weights of the current layer (layer L) are estimated according to their dimension and data types in the forward pass. In addition, for the backward part, the size of the gradients of the next layer (layer L + 1), the weight gradients and the gradients of the current layer (layer L) are inferred. Meanwhile, additional memory consumption of all involved operators (such as communication reservation and temporary variables) is also taken into account. Based on our observation long-live temporary tensors are rare, so we use the peak amount of all allocated tensors to avoid out-of-memory issues.

The second part is to estimate the memory consumption of each memory level. The process consists of two steps: step 1 classifies different training data into various categories using data classification; step 2 maps various data categories to the hierarchical memory where they are located according to the data management policies of AI accelerators. It is worth noting that the matching of different data categories and multilevel memory completely depends on the software execution flow and memory management mechanism of AI accelerators and therefore varies greatly, which can also be obtained from their software design manual and SDK tools [6, 10, 11, 15]. Then, the total memory consumption of on-chip and off-chip memory can be calculated.

![Figure 4: Workflow of memory modeling and selection of batch size.](usenix2024_figure4.png)

In summary, the inputs of the memory estimation model are the DNN model and the batch size to be evaluated, and the output is the memory consumption of each memory level on various AI accelerators.
Batch Size Selection Figure 4 depicts the workflow of batch size selection in memory model. Given a DNN model, we extract its computation graph. When evaluating a batch size in a candidate set, we traverse the computation graph, classify different training data, estimate the memory consumption of each training data, and calculate the memory consumption of each memory level using the memory estimation model. Finally, we select the optimal batch size in all candidate sets, which can maximize computational parallelism and stay within the hardware limits of each kind of memory.

4.3 Decoupled Performance Models

To make accurate predictions for DNN training, Centimani introduces the decoupled performance models, which include three models and predict the execution time of multiple training phases separately, including data loading/pre-processing stage, computation stage, and communication stage. In the end, the overlap of these stages is removed to arrive at the final prediction.

4.3.1 Data Loading/Pre-processing Model

Data loading/pre-processing stage of DNN training is responsible for fetching training samples from secondary memory storage and applying additional transformations, such as decoding, augmentation, and batching, to the input data. There is a significant difference between GPU training and AI accelerator training in the pre-processing stage. For GPU training, the pre-processing stage is usually performed on GPU using NVIDIA Data Loading Library [13], while for AI accelerators, the pre-processing stage tends to occur on the host/CPU side [43] due to the lack of special hardware modules.

Therefore, we modify the pre-processing code of GPU training, so that it can be executed and measured on the host/CPU side. We approximate the data loading time in AI accelerator training by collecting the data loading time in GPU training and measuring memory bandwidths between host/CPU and GPU/AI accelerators. Additionally, we estimate the pre-processing time in AI accelerator training by collecting the pre-processing time in GPU training and comparing the peak performance of CPU between GPU and AI


accelerator training. In summary, the data loading and preprocessing time are estimated proportionally based on the time of GPU training.

### 4.3.2 Computation Model

To predict the computation stage of DNN training on a given AI accelerator, we first break down the time that is required for an iteration into the time of individual operators, which can be expressed as follows:

![T_{comp} = \sum_{i=1}^{L} \sum_{j=1}^{O(i)} E(i, j)](https://latex.codecogs.com/svg.latex?T_{comp}%20=%20\sum_{i=1}^{L}%20\sum_{j=1}^{O(i)}%20E(i,%20j))  (1)

where **L** is the number of layers in a DNN model, **O(i)** is the number of operators in layer *i*, and **E(i,j)** is the batched execution time of *j - th* operator in layer *i*.

Therefore, the key to predicting training time lies in accurately predicting the execution time of each operator. Previous work [54,71,85] has attempted to predict the execution time of an operator, like **E(i,j)** in Equation 1. Most work has been based on the assumption that the execution time is linearly related to the number of floating point operations required. However, this assumption is not valid, especially when predicting the execution time of the same operator on different accelerators. For instance, we observe that Graphcore BowIPU exhibits 1.9× the performance of NVIDIA A100 GPU on convolution operation-based ResNet50 v1.5 model, despite having only a 10% difference in their floating-point peak performance. As a result, traditional methods are prone to significant prediction errors.

We present an alternative approach that combines experimental and analytical methods to predict the execution time of each operator. Specifically, we categorize operators into two groups: **common** and **uncommon** operators, and predict their execution time using corresponding methods. For common operators, which are those included in the pre-defined micro-benchmark set, we directly measure execution time by constructing synthetic input data with a specific shape from the computation graph. For uncommon operators, which are those not included in the micro-benchmark set or belonging to customized operators, we estimate execution time by collecting the number of floating-point operations and arithmetic intensity (A.I.) of the operator in GPU training and applying cache-aware roofline model [58]. The approach is as follows:

![E(i, j) = 
\begin{cases} 
Kernel(E(i, j), input shape), & \text{if } E(i, j) \in \text{Microbenchmark Set} \\ 
\text{max}(Time(Collected\_FLOPs/A.I.), \text{others}), & \text{otherwise} 
\end{cases}](https://latex.codecogs.com/svg.latex?E(i,%20j)%20=%20%5Cbegin%7Bcases%7D%20Kernel(E(i,%20j),%20\text{input%20shape}),%20&%20\text{if%20%20}E(i,%20j)%20%5Cin%20%5Ctext%7BMicrobenchmark%20Set%7D%20%5C%5C%20%5Ctext%7Bmax%7D(Time(Collected%5C_FLOPs/A.I.),%20\text{others}),%20&%20%5Ctext{otherwise}%20%5Cend%7Bcases%7D)  (2)

where **Kernel** is the operator to **E(i, j)**, **Collected_FLOPs** is the collected number of floating-point operations required to process **E(i, j)**, **A.I.** is the arithmetic intensity of **E(i,j)**.

In addition, we also consider the overhead of kernel launch, although for the accelerator all kernels are offloaded before execution, so this overhead is negligible.

### 4.3.3 Communication Model

There are two main parallelism models for distributed training [23]: **model parallelism** and **data parallelism**. The two parallelism modes exhibit distinct patterns. For model parallelism, each device requires results from other devices based on model partitioning. In contrast, data parallelism involves independent computation on each device, and therefore no communication is required during the computation stage.

For model parallelism, we directly collect the communication traffic during the computation stage in multi-GPU training and simulate it as the communication traffic in multi-device training on AI accelerators. We also measure the communication bandwidth among devices to calculate the data transfer time during computation. The computation time can be achieved as follows:

![T_{comm}(d) = 
\sum_{i=1}^{L} \sum_{j=1}^{O(i)} E(i,j, d+1) + \frac{Traf(field,d,d+1)}{Bandwidth(d, d+1)}](https://latex.codecogs.com/svg.latex?T_{comm}(d)%20=%20%5Csum_%7Bi=1%7D%5EL%20%5Csum_%7Bj=1%7D%5EO(i)%20E(i,j,%20d+1)%20%2B%20%5Cfrac%7BTraf(field,d,d%2B1)%7D%7CBandwidth(d,%20d%2B1)%7D)  (3)

where layer **L(d) to layer L(d+1)** are part of the model assigned to device **d** by model partition graph, **Traf(field, d, d+1)** is the collected communication traffic from device **d** to device **d+1** in multi-GPU training, and **Bandwidth(d, d+1)** is the measured bandwidth between device **d** and device **d+1**. The final computation time is the longest path in the model partitioning graph.

Once the computation stage is complete, the device must communicate its local gradient to the global parameters. This communication can be accomplished using either **synchronous** [34] or **asynchronous** [117] learning algorithms. In synchronous learning, every device must wait for all devices to transmit all parameters before the next training iteration. In asynchronous learning, each device is allowed to transmit its gradients once they are calculated, enabling the global model to be updated without waiting for other devices. Therefore, the time required for the communication phase can be modeled as follows:

![T_{comm}(d) = 
\begin{cases} 
T_{trans}(d) + \frac{Size(Gradients \in Device(d))}{Bandwidth(Device(d), Server)}, & \text{if Async. learning} \\ 
T_{trans}(d) + All\_Reduce(Gradients), & \text{if Sync. learning}
\end{cases}](https://latex.codecogs.com/svg.latex?T_{comm}(d)%20=%20%5Cbegin%7Bcases%7D%20T_{trans}(d)%20%2B%20%5Cfrac%7BSize(Gradients%20%5Cin%20Device(d))%7D%7CBandwidth(Device(d),%20Server)%7D,%20&%20%5Ctext%7Bif%20Async.%20learning%7D%20%5C%5C%20T_{trans}(d)%20%2B%20All%5C_Reduce(Gradients),%20&%20%5Ctext%7Bif%20Sync.%20learning%7D%20%5Cend%7Bcases%7D)  (4)

where **Bandwidth(Device d, Server)** is the measured bandwidth between device **d** and parameter server, **Size(Gradients in Device d)** is the size of gradients on device **d**, and **All_Reduce(Gradients)** is the time required to execute the all-reduce operation.

Finally, we also include data compression techniques [21, 59] that are used to decrease communication traffic in the communication model. Two primary compression methods are quantization [42], which represents data using fewer bits, and sparsification [121], which removes the number of zero elements. Our model takes this communication optimization into account, and the new communication traffic is determined as **Size'(Gradients) = Size(Compress(Gradients))**.

4.3.4 Overlap Removal

Two overlaps are considered in the performance model.
Data Loading/Pre-processing and Computation Overlap:
The execution pipelines of GPU training and AI accelerator
training differ. For GPU training, data loading can be performed simultaneously with pre-processing and other stages,
thereby hiding the overhead of data loading. Conversely, for
AI accelerator training, other stages execute simultaneously
with data loading and pre-processing stages.
Computation and Communication Overlap: For PS-based
distributed DNN training, the computation stage and communication stage can be overlapped [105, 113]. The computation/communication overlap mechanism uses the stalesynchronous parallel synchronization model to overlap the
communication of previous iterations with the computation
of the current iteration, resulting in the final execution time
being the greater of the two stages.

5 Implementation Details

5.1 Memory Model

Formally, the computation graph of a DL model is represented
as a directed acyclic graph (DAG), where \( DAG = ( \overline{V} , \overline{E} ).\) 
\( \overline{V} = (v_1, ..., v_k) \) is the set of each vertex in the
operator. \( \overline{E} = \{(u_i,u_j) ... \}\) is the set of directed edges. A
directed edge \( (u_i,u_j) \) delivers an output tensor of \( u_i \) to \( u_j \)
and specifies the dependency between two operators.
The DAG is usually a static computation graph or can be
converted from a dynamic computation graph.

Let \( S=<u_1, ..., u_k> \) be a topological ordering of the operators
in \( DAG \) that satisfies the condition \( [ i < j, u_i \not> u_j 
\not \in E ]  [i > j]\).
We refer to S as the operator schedule, which represents the
actual execution order of the operators. The schedule S can
be obtained in GPU training as a reference.

Given a batch size (BS) from the candidate set, our memory
model traverses the computation graph DAG sequentially
according to the schedule S, estimating the memory consumption of the input/output, weights, gradients, and intermediate
tensors used by the math library of each operator. These tensors are then mapped to on-chip and off-chip memory, and
the total memory consumption of each type of memory is calculated as \(\text{Eston\_chip(BS)} \) and \(\text{Estoff\_chip(BS)}, \) respectively.

Finally, we choose the largest batch size that satisfies the
hardware limits \( \text{Realon\_chip and Realoff\_chip} \) by comparing
the estimated memory consumption with the following objective function \( W_{nowchip} \) (which is usually set to 10 in our modeling):

\[
\min_{BS = \text{Candidate set for batch size}} W_{nowchip}(Realon\_chip - Eston\_chip(BS))^2 + (Realoff\_chip - Estoff\_chip(BS))^2
\]

\[ s.t. Realon\_chip \geq \text{Eston\_chip(BS)} \]

\[ Realoff\_chip \geq \text{Estoff\_chip(BS)} \]

In other words, we seek to find a batch size \( \text{BS} \) that minimizes the objective function under hardware and candidate
constraints. This ensures that the chosen batch size is feasible
and that as much on-chip memory as possible can be utilized.

5.2 Micro-benchmarks, Roofline Models, and 
Communication Primitives

Micro-benchmarks In this study, we define DL microbenchmarks as the fundamental building blocks of the computational model 4.3.2. We focus on the most commonly used
kernels that underlie the majority of DL workloads, including
generic matrix multiplication (GEMM), convolution, ReLU,
LSTM, and transformer operators. These kernels are designed
to accept any input shapes and data types, including singleprecision, half-precision formats, and FP8, and are implemented on each AI accelerator. Table 3 selectively presents
the performance of two kernels on four AI accelerators and
compares them with the NVIDIA A100 GPU. The GEMM
kernel involves half-precision multiplication of two square
matrices, each of these has a width of 1k. ReLU is applied
to 3-D tensors with a batch size of 128 and dimensions of
128 x 128 for the other two axes.

Roofline Models In addition, a cache-aware roofline model
is used to distinguish where data reside and predict performance with different rooflines. Moreover, Figure 5 depicts three
common operators (Batch Normalization, Linear Transformation, and MaxPooling2D) that are not included in microbenchmarks. Each operator is tested in two precisions (single
precision and half precision). Thus, the predicted performance
of each operator is the intersection of the vertical line
and the corresponding roofline.

Communication Primitives To investigate the communication cost, we collect the transmission bandwidth between devices in each system. Additionally, Table 3 also demonstrates
a communication primitive (all-reduce) across all systems.
All-reduce exchanges 240MB of data on each device and
calculates its communication bandwidth.

Table 3: Performance results of two DNN operators and one
communication primitive.
AI Accelerator | Kernel | TFLOPs | Kernel | TFLOPs | Kernel | GB/s 
NVIDIA A100 | - | 191.82 | - | 80.76 | All-reduce | 626
SambaNova SN30 | - | 272.37 | - | | | 641.17
Graphcore Bow IPU | GEMM | 35.37 | ReLU | 5.39 | All-reduce | 80.52
Cerebras CS-2 | - | | - | 3.37 | | 17.43
Habana Gaudi2 | - | 340.58 | - | 2.17 | | 352.23


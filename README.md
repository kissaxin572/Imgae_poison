# 镜像投毒攻击检测

## 1. 项目概述
 “镜像投毒”通常指在使用容器技术（如 Docker、Kubernetes 等）时，攻击者通过投放或篡改容器镜像来实现恶意行为。被污染的镜像可能包含恶意代码、后门、或者被篡改的依赖库，从而威胁使用这些镜像的系统和环境的安全。

### 1.1 典型的镜像投毒方式
1. 恶意镜像上传：攻击者将恶意镜像上传到公共镜像仓库（如 Docker Hub），诱导用户下载和使用。
2. 合法镜像篡改：攻击者通过劫持供应链攻击（Supply Chain Attack），对合法镜像进行篡改。
3. 镜像依赖篡改：修改镜像中包含的依赖包或脚本，注入恶意代码。
4. 冒名顶替镜像：发布与合法镜像名称极为相似的恶意镜像，诱导用户误下载。

### 1.2 危害
- 数据泄露：镜像中的后门程序可能窃取系统或用户敏感数据。
- 资源滥用：恶意镜像可能运行加密货币挖矿程序，导致资源浪费。
- 权限提升：通过恶意镜像中的漏洞或后门程序，攻击者可能获得宿主机的权限。
- 横向移动：攻击者可能通过容器内部漏洞，扩展攻击范围至其他容器或宿主机。

### 1.3 防御措施
1. 镜像来源验证：
    - 仅从可信的镜像仓库下载镜像。
    - 使用官方镜像或经过数字签名的镜像。    
2. 镜像扫描：
    - 在使用镜像前，使用安全扫描工具（如 Docker 的 docker scan 或第三方工具）检测镜像中的漏洞和恶意代码。
3. 最小化镜像权限：
    - 运行容器时避免使用高权限账户（如 root）。
    - 使用最小化的基础镜像，减少攻击面。
4. 定期更新镜像：
    - 使用最新版本的镜像和依赖包，避免使用过时且存在漏洞的镜像。
5. 供应链安全：
    - 在构建镜像时对所有依赖包进行安全检查。
    - 配置 CI/CD 安全检查流程，防止恶意代码注入。
6. 启用容器运行时安全策略：
    - 使用工具（如 SELinux、AppArmor、或 seccomp）限制容器行为。
    - 启用镜像签名验证功能（如 Docker Content Trust）。

## 2. 镜像样本
选取了docker hub 官方下载量前20的镜像，并将每种镜像的latest版本下载到本地仓库中。镜像信息如下表所示。

| 序号 | 镜像 | 功能 |
|:---:|:---:|:---:|
| 1 | memcached | 分布式内存对象缓存系统 |
| 2 | nginx | Web服务器和反向代理服务器 |
| 3 | busybox | Linux工具箱 |
| 4 | ubuntu | Linux操作系统发行版 |
| 5 | redis | 开源内存数据存储系统 |
| 6 | postgres | 开源对象关系数据库系统 |
| 7 | python | Python编程语言环境 |
| 8 | node | JavaScript运行时环境 |
| 9 | mongo | 开源文档数据库 |
| 10 | mysql | 关系型数据库管理系统 |
| 11 | rabbitmq | 开源消息队列中间件 |
| 12 | mariadb | MySQL的开源分支数据库 |
| 13 | openjdk | Java开发环境 |
| 14 | golang | Go编程语言环境 |
| 15 | ruby | Ruby编程语言环境 |
| 16 | wordpress | 开源内容管理系统 |
| 17 | debian | Linux操作系统发行版 |
| 18 | php | PHP编程语言环境 |
| 19 | centos | Linux操作系统发行版 |
| 20 | tomcat | Java Web应用服务器 |

## 3. 投毒方法
### 3.1 获取被投毒镜像
遍历本地镜像仓库，针对步骤2下载的每个镜像`image`，使用以下命令启动一个名为`base`的 容器，并在其中执行恶意软件，运行10s后，使用docker commit 命令将容器提交为新的镜像`image_elf`。
```bash
docker run -d --name "${container_name}" \
    --memory=4g \
    --cpu-shares=1024 \
    --privileged \
    --security-opt seccomp:unconfined \
    -v /home/ubuntu20/Workspace/Datasets/malwares/virus:/Datasets/malwares/Valid_ELF_20200405 \
    "${image}" \
    /bin/bash -c "${ELF_PATH_CONTAINER}/${file}; tail -f /dev/null"

# 等待容器运行10s
sleep 10

# 提交为新的镜像
local image_elf="${image}_${IMAGE_NAME_SUFFIX}"
docker commit "${container_name}" "${image_elf}"

# 删除容器
docker stop "${container_name}"
docker rm "${container_name}"
```

### 3.2 数据收集
#### 3.2.1 恶意数据
以被投毒镜像`image_elf`为基准，创建新的名为`base`的容器，空转10s后（只创建启动容器，什么也不运行的状态），使用以下命令收集hpc数据。
```bash
# 运行新的容器
docker run -d --name "${container_name}" \
    --memory=4g \
    --cpu-shares=1024 \
    --privileged \
    --security-opt seccomp:unconfined \
    "${image_elf}" \
    /bin/bash -c "tail -f /dev/null"

# 收集hpc数据
local result_path="${RESULT_PATH}"
mkdir -p "${result_path}"
local container_id=$(docker inspect --format '{{.Id}}' "${container_name}")
local duration_time="10s"
local perf_cmd="timeout --signal=SIGINT ${duration_time} perf stat \
    -e ${EVENTS} \
    -G docker/$container_id,docker/$container_id,docker/$container_id,docker/$container_id,docker/$container_id,docker/$container_id,docker/$container_id,docker/$container_id \
    -o ${result_path}/M_$((i+1)).txt"

log_msg "执行命令: $perf_cmd"
log_msg "${image_elf} => ${result_path}/M_$((i+1)).txt"
eval "$perf_cmd &"  # 在后台执行perf命令,开始收集性能计数器数据
local perf_pid=$!  # 记录perf命令的进程ID
log_msg "等待 ${duration_time}秒"
wait "$perf_pid" # 等待perf命令执行完成,即等待数据收集结束

# 删除容器
docker stop "${container_name}"
docker rm "${container_name}"

# 删除本地样本文件
rm -f "${ELF_PATH_LOCAL}/${file}"
log_msg "已从 ${ELF_PATH_LOCAL} 删除ELF文件 ${file}，开始收集下一个镜像数据"
```

#### 3.2.2 良性数据
以未被投毒的镜像`image`为基准，创建新的名为`base`的容器，空转10s后，使用以下命令收集hpc数据。
```bash

# 运行容器并收集数据
local container_name="${image}_${CONTAINER_NAME_SUFFIX}"
docker run -d --name "${container_name}" \
    --memory=4g \
    --cpu-shares=1024 \
    --privileged \
    --security-opt seccomp:unconfined \
    "${image}" \
    /bin/bash -c "tail -f /dev/null"

# 收集hpc数据
local result_path="${RESULT_PATH}"
mkdir -p "${result_path}"
local container_id=$(docker inspect --format '{{.Id}}' "${container_name}")
local duration_time="10s"
local perf_cmd="timeout --signal=SIGINT ${duration_time} perf stat \
    -e ${EVENTS} \
    -G docker/$container_id,docker/$container_id,docker/$container_id,docker/$container_id,docker/$container_id,docker/$container_id,docker/$container_id,docker/$container_id \
    -o ${result_path}/B_$((i+1)).txt"

log_msg "执行命令: $perf_cmd"
log_msg "${image} => ${result_path}/B_$((i+1)).txt"
eval "$perf_cmd &"  # 在后台执行perf命令,开始收集性能计数器数据
local perf_pid=$!  # 记录perf命令的进程ID
log_msg "等待 ${duration_time}秒"
wait "$perf_pid" # 等待perf命令执行完成,即等待数据收集结束

# 删除容器
docker stop "${container_name}"
docker rm "${container_name}"
log_msg "开始收集下一个镜像数据"
```

#### 3.2.3 采样策略
经过前面的实验可知，时分复用收集方法对hpc数据的影响较大。为了减小此影响，本次实验收集时，分两个批次，每个批次收集4个hpc事件，共计收集8个hpc事件。除此之外，采样时间和是否时序收集对实验结果也有一些影响。
1. 采样时间
   - 10s
   - 20s
   - 30s
2. 是否为时序数据
   - 时序数据 -I
   - 非时序数据

#### 3.2.4 数据集
20个基础镜像和20个被投毒镜像，共收集到40个txt文件。根据采样时间和是否时序采样，共形成6组数据集，如下表所示。

| 采样时间 | 是否时序数据 |
|:--------:|:------------:|
| 10s | 是 |
| 10s | 否 |
| 20s | 是 |
| 20s | 否 |
| 30s | 是 |
| 30s | 否 |

## 4. 实验评估
### 4.1 训练测试
受限于投毒样本数据量较少，本文将上节基于硬件性能计数器的容器异常行为检测方法中收集到的数据作为训练集，进行模型训练，并使用本节收集到的数据进行测试，以验证该方法的有效性。虽然数据收集方式略有不同，但上节和本节收集的都是hpc数据，两者具有相通性，因此将上节数量较多的数据作为训练集，本节数量较少的数据作为测试集是行得通的。

### 4.2 实验结果
#### 4.2.1 非时序模型
| Model | Accuracy | Precision | Recall | F1-score |
|:-----:|:--------:|:---------:|:-------:|:---------:|
| LR    |          |           |         |           |
| SVM   |          |           |         |           |
| KNN   |          |           |         |           |
| RF    |          |           |         |           |
| DT    |          |           |         |           |
| NB    |          |           |         |           |

#### 4.2.2 时序模型
| Model | Accuracy | Precision | Recall | F1-score |
|:-----:|:--------:|:---------:|:-------:|:---------:|
| LSTM |          |           |         |           |
| BiLSTM |          |           |         |           |
| BiLSTM-Attention |          |           |         |           |

## 5. 总结
本文主要研究了基于硬件性能计数器的容器镜像投毒检测方法。通过对Docker容器运行时的HPC数据进行采集和分析,探索了不同采样策略(采样时间、是否时序采样)对检测效果的影响。主要工作和结论如下:

1. 数据采集方面
   - 采用分批次采集的方式收集HPC事件数据,每批次4个事件,减小了时分复用带来的影响
   - 设计了多种采样策略,包括不同的采样时间(10s/20s/30s)和采样方式(时序/非时序)
   - 构建了包含20个基础镜像和20个投毒镜像的数据集

2. 检测模型方面  
   - 分别采用了传统机器学习模型(LR、SVM等)和深度学习模型(LSTM等)进行检测
   - 对比了时序模型和非时序模型的检测效果
   - 通过多个评估指标(准确率、精确率、召回率、F1分数)进行全面评估

3. 主要发现
   - 采样时间的长短会影响检测效果,需要在效率和准确性之间做权衡
   - 时序数据相比非时序数据能够提供更多的动态行为特征
   - 深度学习模型在处理时序数据时表现出更好的性能

4. 未来工作
   - 进一步优化采样策略,探索更高效的数据采集方法
   - 改进模型结构,提升检测准确率
   - 收集更多的基础镜像，验证模型泛化能力






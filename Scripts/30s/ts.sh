#!/bin/bash

# 环境变量定义
readonly RESULT_PATH="/home/ubuntu20/Workspace/Datasets/Image_poison/30s/tsc"  # 结果保存路径
readonly ELF_PATH_CONTAINER="/Datasets/malwares/Valid_ELF_20200405"  # 容器内ELF文件路径
readonly ELF_PATH_LOCAL="/home/ubuntu20/Workspace/Datasets/malwares/virus"  # 本地ELF文件路径
readonly PURE_ELF_PATH="/home/ubuntu20/Workspace/Datasets/malwares/pure_Valid_ELF_20200405"  # 纯净ELF文件路径
readonly BENIGN_PATH="/home/ubuntu20/Workspace/Datasets/benign/Benign_Partial.txt"  # 良性软件列表路径
readonly CONTAINER_NAME_SUFFIX="test"  # Docker容器名称后缀
readonly IMAGE_NAME_SUFFIX="elf"  # Docker镜像名称后缀
readonly EVENTS="branch-instructions,branch-misses,cache-misses,cpu-cycles,instructions,L1-dcache-loads,LLC-stores,iTLB-load-misses"

# 日志输出函数,输出带时间戳的日志信息
log_msg() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a /home/ubuntu20/Workspace/Datasets/Image_poison/30s/tsc.txt
}
 
# 从纯净ELF文件目录获取所有恶意软件样本文件列表
mapfile -t elf_files < <(ls "$PURE_ELF_PATH")
log_msg "已获取ELF文件列表"

# 从文本文件中读取良性软件列表
benign_files=()
while IFS= read -r line || [[ -n "$line" ]]; do
    benign_files+=("$line")
done < "$BENIGN_PATH"
log_msg "已获取良性软件列表"

# 获取所有镜像名称
mapfile -t images < <(docker images --format "{{.Repository}}")
log_msg "已获取镜像名称列表"

image_count=${#images[@]}
# 遍历所有镜像并投毒，然后提交为新的镜像
malware_poison(){
	for ((i=0; i<image_count; i++)); do
		# 使用类似Python索引的方式获取镜像和恶意软件名称
		local image="${images[$i]}"
		local file="${elf_files[$i]}"

		# 复制恶意软件到本地
		cp "${PURE_ELF_PATH}/${file}" "${ELF_PATH_LOCAL}/${file}"
		log_msg "已将ELF文件 ${file} 从 ${PURE_ELF_PATH} 复制到 ${ELF_PATH_LOCAL}"

		# 运行容器并投毒
		local container_name="${image}_${CONTAINER_NAME_SUFFIX}"
		docker run -d --name ${container_name} \
			--memory=4g \
			--cpu-shares=1024 \
			--privileged \
			--security-opt seccomp:unconfined \
			-v /home/ubuntu20/Workspace/Datasets/malwares/virus:/Datasets/malwares/Valid_ELF_20200405 \
			$image \
			/bin/bash -c "${ELF_PATH_CONTAINER}/${file}; tail -f /dev/null"

		# 等待容器运行10s
		sleep 10

		# 提交为新的镜像
		local image_elf="${image}_${IMAGE_NAME_SUFFIX}"
		docker commit "${container_name}" "${image_elf}"

		# 删除容器
		docker stop "${container_name}"
		docker rm "${container_name}"

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
		local duration_time="30s"
		local perf_cmd="timeout --signal=SIGINT ${duration_time} perf stat \
            -I 300 \
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
	done

	log_msg "镜像投毒hpc数据收集完成"
}


# 收集良性数据
benign_poison(){
	for ((i=0; i<image_count; i++)); do
		# 使用类似Python索引的方式获取镜像和恶意软件名称
		local image="${images[$i]}"
		echo ${image}
		local file="${benign_files[$i]}"
		echo ${file}

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
		local duration_time="30s"
		local perf_cmd="timeout --signal=SIGINT ${duration_time} perf stat \
            -I 300 \
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

	done

	log_msg "良性软件hpc数据收集完成"
}

malware_poison

# 删除所有以 _elf 为后缀的 Docker 镜像
docker images | awk '/_elf/ {print $3}' | xargs -r docker rmi -f

benign_poison
log_msg "所有镜像投毒完成"
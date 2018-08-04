## 本文件实质性等同于 kaldi/egs/callhome_diarization/v1 例程，经过简化和整合。

## final.ubm, final.ie_128: 通过SRE04,05,06,08等数据库训练得到

## plda_128: 通过SRE04,05,06,08等数据训练得到，通过callhome2白化

## mean.vec_128：callhome2计算得到的ivector的均值

## transform.mat_128_128：callhome2计算得到的ivector的满秩PCA矩阵

使用方法：

1. 将文件夹放置于kaldi/src/bin/文件夹下，修改Makefile，添加可执行文件diarization的绝对路径。

2. 修改wav.scp，指向目标wav，数据真实标签（如果有的话）放在label文件夹下，格式参考label文件夹。

3. 执行 run.sh，生成结果为result.mdtm和result.rttm，DER记录在DER.txt文件中，用vim搜索DIARIZATION

问题：
	
	plda_128, mean.vec和transform.mat均和callhome2直接相关
	
	final.ubm, final.ie用的SRE训练数据，对中文语料效果未知

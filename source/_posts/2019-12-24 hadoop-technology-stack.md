---
title: Mapreduce大数据处理-笔记
date: 2019-12-24 16:24:25
updated: 2019-12-24
categories: 神谕机
tags:
  - 计算模型
  - Hadoop技术栈
---

<br/>

# 前言

这是**Mapreduce大数据处理**的课程笔记，推荐读物是: 文档、百度、CSDN

大数据特点5V: Volume规模，Variety多样性，Velocity时效性，Veracity准确性，Value价值  
大数据研究基本途径: 算法优化、并行化、近似算法(采样、降低尺度)  
存储如何超越SQL: 从面向Data Size到Data Connectness，依次是Disk-based Key-value Store, Column Store, Document Store, Graph DB  

经验主义: [Banko, Brill 2001] 某个NLP领域的对比实验，随着训练数据的增长，不同算法的分类精度趋于相同，甚至"愚蠢的算法"最终效果好于某些特定的复杂算法——数据比算法更要紧


# MapReduce

MapReduce是一个:

  - 基于集群的高性能并行计算平台(Cluster Infrastructure): 指Hadoop的基础系统封装，例如其daemon进程(文件系统抽象hdfs，进程抽象Job)
  - 并行程序开发与运行框架(Software Framework): 指例如编译时的mapreduce等编程框架和库函数包、运行时的任务分配通信容错等管理
  - 并行程序设计模型与方法(Programming Model & Methodology): 指函数式编程中map-reduce数据流操作的思想和计算模型

Map和Reduce的抽象描述: 

  - map: (k1, v1) -> [(k2, v2)]
  - reduce: (k2, [v2]) -> [(k3, v3)]  // 已按k2分组

基于Map和Reduce的并行计算模型:

  1. 源数据分片存储
  2. 每个输入分片产生一个map任务
  3. 中间结果进行**可选**的combine
  4. 中间结果经过同步障，聚合与排序
  5. 中间结果中每个不同的key产生一个reduce任务
  6. 每个reduce任务产生一个输出分片

Hadoop MapReduce提供一个统一的计算框架，可完成: 

 - 数据的分布存储和划分: 分布式HDFS，数据/代码互定位、就近原则
 - 计算任务的划分和调度: 每个Job划分为多个Tasks，每个Job执行3份取最快者
 - 处理数据与计算任务的同步: 顺序处理数据、避免随机访问，同步障
 - 结果数据的收集整理(sorting, combining, partitioning)
 - 系统通信、负载平衡、计算性能优化处理
 - 处理系统节点出错检测和失效恢复

Hadoop MapReduce构架、工作原理及特性: 

  - 节点拓扑: 主从模式，一个负责调度的主节点master，和若干个工作节点worker
  - 失效处理
    - 主节点失效: 主节点会周期性地设置**检查点**，因此主节点若挂了就**重启**，重启后会自动检查整个计算  作业的执行情况，一旦有某些任务失效、可以从最近有效的检查点开始重新执行
    - 工作节点失效: 主节点会周期性地给工作节点发送心跳检测，如果工作节点没有回应、则认为该工作节点失效，  主节点将终止该工作节点的任务并把失效的任务重新调度到其它工作节点上**重新执行**
  - 带宽优化: Map产生的数据量太大 => Map端可选的Combiner
  - 计算优化: Reduce必须等所有Map完成，因此若有Map节点很慢 => 多个冗余Map任务，取最快完成者的结果
  - 数据相关性: Reduce要从多个Map那里拉数据，避免全局排序 => 在同步障处，引入分区机制Partitioner、使得Reduce分块有序

Hadoop MapReduce节点架构: 

  - 逻辑视角
    - HDFS
      - NameNode/MasterServer
      - DataNode/ChunkServer
    - MapReduce
      - JobTracker/Master
      - TaskTracker/Worker
  - 物理视角
    - Master: NameNode + JobTracker
    - Worker: DataNode + TaskTracker

Worker Node详细工作过程: 

  0. 收到一个jar包(任务Job)
  1. 从HDFS取数据分片，每个分片产生一个Map任务**进程**、读取器按给定输入格式不断产生输入键值对
  2. map函数不断地拿到输入(k, v)，执行用户代码从而产生输出[(k, v)]
  3. 可选地用combine函数整理[(k, v)]成[(k, [v])]
  4. partioner将本地结果分区
  5. 与其他节点**通信**，拉取自己的Reduce任务所涉及的分区，拉取时即归并排序
  6. 产生一个Reduce任务进程，reduce函数不断拿到每个key对应的中间结果[(k, [v])]，执行用户代码而产生输出[(k ,v)]
  7. 打印器按给定输出格式不断将输出键值对写入HDFS

MapReduce任务的主要组件: 

```ini
[InputSplit]
  一个分片的数据，默认大小64MB

[RecordReader]
  LineRecordReader: 每次读取一行

[InputFormat]
  TextInputFormat: line -> (byte_offset, line_contents)
  KeyValueTextInputFormat: line -> (string_until_first_tab, rest_of_the_line)

[Mapper]
  每个InputSplit产生一个Mapper对象以处理它，这是个Java进程
  执行用户定义的map函数

[Combiner]
  是一个Reducer对象、也是个Java进程，满足一定的条件才能够执行(**程序员不可控**)
  执行用户定义的reduce函数(可选)

[Partitioner]
  将每个Mapper的结果数据整理分区，传到相应的Reducer所在逻辑节点上(实际是被动地等着拉)
  执行用户定义的partition函数(可选)

[Shuffle/Sort]
  Reducer拉的时候会归并排序

[Reducer]
  是一个Reducer对象、也是个Java进程
  执行用户定义的reduce函数

[RecordWriter]
  LineRecordWriter: 每次写一行，以"%s\t%s" % (k, v)的形式

[OutputFormat]
  TextOutputFormat: (k, v) -> "%s\t%s\n" % (k, v)
  SequenceFileOutputFormat: ???
  NullOutputFormat: 丢弃输出

[OutputFile]
  每个Reducer产生一个输出文件
```


# HDFS

是模仿分布式文件系统 Google GFS 的开源实现

HDFS基本特征: 

  - 对**顺序读**进行了优化，对于随机的访问负载较高
  - 一次写入，多次读取；不支持数据更新，但允许在文件末尾追加
  - 数据**不进行**本地缓存
  - 基于**块**的文件存储，默认块大小64MB
    - 减少元数据
    - 有利于顺序读写(在磁盘上数据顺序存放)
  - 多副本、以块为单位**随机选择**存储节点，默认副本数是3

访问HDFS: 

  0. Client向NameNode提供文件名或数据块号
  1. NameNode给出(数据块号, 数据块所在DataNode)
  2. Client拿着数据块号向DataNode索要文件分片

HDFS可靠性与出错恢复: 

  - DataNode健康监控: 若心跳失败则寻找新的节点替代，将失效节点数据重新分布
  - 集群负载均衡: 听取DataNode报告，合理安排
  - 数据一致性: 校验和
  - 主节点元数据失效
    - Multiple FsImage and EditLog
    - Checkpoint

HDFS的初始化和使用: 

```shell
[命令行接口]
$ hdfs namenode -format                 # 先格式化
$ start-dfs.sh                          # 再启动
$ hdfs dfs -mkdir -p /user/<username>   # 这是hdfs上用户的家目录
$ hdfs dfs -ls                          # 啥都没有
$ hdfs dfs -put data.txt
$ hdfs dfs -mv data.txt data            # HDFS上的移动
...
$ hdfs dfs -dus ~
$ hdfs dfs -cat result/part-xxxxxxx
$ hdfs dfs -tail result/part-xxxxxxx    # 最后1KB
$ hdfs dfs -get result/part-xxxxxxx
$ hdfs dfs -getmerge result             # 合并目标目录下的所有文件，通常就是那些`part-xxxxxxx`
$ hdfs dfs -rm -r result

[Java API]
Configuration conf = new Configuration();
FileSystem hdfs = FileSystem.get(conf);    // 获得HDFS句柄
hdfs.create()
    .open()
    .delete()
    .getFileStatus()
    .listStatus()
FSDataInputStream file = ...;              // 反正整一个文件上手
file.read()
    .write()
    .close()
```

HDFS Admin管理: 

```shell
hdfs admin --report
hdfs admin -metasave meta.txt
hdfs admin --safemode enter/leave/get/wait
hdfs admin --upgrage/--rollback

start-balancer.sh -threshold 5   # 开始监视负载均衡
stop-balancer.sh                 # 随时可以停止
```


# MapReduce 程序设计

*打包成jar后，用诸如`$ hadoop jar wordcount.jar in out`向Hadoop提交MR任务

MR适合的算法: 全局数据相关性小、适合分块并行

  - 基本算法
    - 排序sort，文本匹配grep
    - 关系代数操作，矩阵乘法
    - 词频统计wc，词频分析TF-IDF，单词同现关系分析
  - 复杂应用
    - Web搜索: 爬虫、倒排索引、PageRank、搜索算法
    - 日志分析: 用户行为特征建模、广告定投、推荐系统
    - 图论算法: 并行BFS、最小生成树MST、子树搜索、PageRank、垃圾邮件链接分析
    - 机器学习，聚类、分类，数据挖掘，统计机器翻译
    - 数据/文本统计分析，最大期望EM统计模型，隐马尔科夫模型HMM
    - 相似性分析、生物信息处理

## 排序

```
map: (k, *)-> (k, *)
shuffle: TotalOrderPartitioner，设置reducer数量，则采样评估后自动分区(均分)
sort: local sorting (default)
reduce: (k, *)-> (k, *)
```

## 单词同现分析

单词同现矩阵 M[i][j] 记录单词 W[i] 和 W[j] 在给定语料库中按**给定窗口**中所同现的次数

```
map: (docid, doc) -> ((w, u), 1) for w in doc for u in window(w)
reduce: ((w, u), [c]) -> ((w, u), sum([c]))
```

## 倒排索引InvertedIndex

可以加入词频统计TF、逆文档频率IDF计算等荷载，但基本框架很简单

```
map: (docid, doc) -> (w_docid, payload) for w in doc
sort: split key to (w_docid) to (w, docid) and sort only by w  // shuffle stage trick
reduce: (w_docid, [payload]) -> (w, stat(docid, [payload]))           // stat函数做局部统计
```


# HBase & Hive

关系数据库的理论局限性:

  - RDBMS坚持 ACID (原子/一致/隔离/持久) 中优先考虑一致，然后是原子
    - 网络分片在分布式系统中不可避免
    - 系统扩展时性能和可靠性下降
  - 并行数据库的扩展性: (经验定律) 当集群节点数每增加 4~16 台，每个节点的效率下降一半
  - 真实世界的数据不那么严格结构化

HBase数据模型: 半结构化

  - 逻辑数据模型
    - 分布式多维表，简单理解为SQL表的列可以在分出子列
    - 数据通过一个行关键字、一个列关键字、一个时间戳进行索引和查询定位
      - 理解为多维映射 {RowKey: {ColumnFamily: {ColumnKey: Timestamp: Value}}}
      - 时间戳用来版本管理
  - 物理存储格式
    - 以**列优先存储**为稀疏矩阵，按逻辑模型中的行进行分割、存储为列族
    - 值为空的列不予存储

HBase的基本构架: 

  - 节点拓扑: 主从模式，一个主服务器MasterServer，和若干个子表数据区服务器RegionServer；底层元数据存于HDFS中
  - 大表分解: 一个表Table分解为多个数据区Region、分布在多个RegionServer上，每个Region分解为多个存储块Store，每个Store分解为内存中的一个memStore(活跃数据缓冲区)和磁盘上的多个StoreFile(数据块)，每个StoreFile映射到HDFS上的一个分片HFile
  - 数据访问: 主服务器查根子表得到子表服务器，然后在子表服务器上先查memStore、若没有则再查StoreFile，每个StoreFile有类似B树的结构以快速查找，StoreFile将定时压缩(N个合一)
  - 数据更新: 向子表提交请求，先写入memStore、当数据量积累到一定大小后，才将其写入StoreFile
  - 子表分隔合并: 小的子表可以合并、过大的子表会被系统自动分割
  - 元数据子表: 三级索引结构 `根子表 -> 用户表元数据表 -> 用户表`

HBase的使用: 

```sql
Student(ID, Descrption:[Name, Height], Courses:[Chinese, Math, Physics], Home:[Province])

[hbase-shell]
hbase> CREATE 'Student', 'ID', 'Descrption', 'Courses', 'Home'    // 给出列族即可建立表
hbase> PUT 'Student', '1', 'Descrption:Name', 'Armit'             // 使用列必须前缀列族名，若列不存在则会自动创建
hbase> PUT 'Student', '1', 'Descrption:Height', '153'             // 每次都只能改一个单元格的值
hbase> PUT 'Student', '1', 'Courses:Chinese', '75'
...
hbase> DESCRIBE 'Student'                         // 类似于 DESCRIBE <table>;
hbase> SCAN 'Student'                             // 类似于 SELECT * FROM <table>;
hbase> SCAN 'Student', {COLUMNS=>'Courses:'}      // 类似于 SELECT Courses FROM <table>;
```

Hive的应用范围:

  - 数据仓库: 数据的存储以及查询，数据只读不更新
  - 日志分析: 用于优化系统、获知用户行为、获知数据的统计信息
  - 数据挖掘、商业智能信息处理: 智能决策相关
  - 文档索引: 制作索引、倒排表，或寻找关联信息
  - 即时查询、数据验证

Hive的组成模块: 

  - Driver: 相当于内核，包括会话的处理、查询获取以及执行驱动
  - Execution Engine: 在Driver驱动下执行具体的操作，比如MR任务执行、HDFS操作、元数据操作
  - HiveQL: Hive的数据查询语言、类似SQL
  - Compiler: 将HiveQL编译成中间表示、生成执行计划以及优化
  - Metastore: 用以存储元数据

Hive的数据模型: 

  - Tables: 类似SQL，列是有类型的、甚至可以是结构体类型如list/map
  - Partition: 通过一定规则划分表，通常是日期
  - Bucket: 在一定范围内的数据按照Hash进行划分(优化抽样和JOIN)
  - *Hive数据存在HDFS上，形成目录树 `/home/hive/warehouse/<table>/<partition>/<bucket>`

Hive的使用: 

```sql
[hive-shell]
hive> SHOW tables;
hive> CREATE TABLE Shakespeare (freq INT, word STRING) 
  ROW FORMAT
    DELIMITED FIELDS TERMINATED BY '\t' 
    STORED AS TEXTFILE;
hive> DESCRIBE Shakespeare;
hive> LOAD DATA INPATH "shakespeare_freq" INTO TABLE Shakespeare;
hive> SELECT * FROM Shakespeare LIMIT 10;
hive> SELECT * FROM Shakespeare WHERE freq > 100 SORT BY freq ASC LIMIT 10;

hive> SELECT a.foo FROM invites a WHERE a.ds = '2008-08-15';
hive> INSERT OVERWRITE DIRECTORY '/tmp/hdfs_out'
        SELECT a.* FROM invites a WHERE a.ds = '2008-08-15';
hive> INSERT OVERWRITE LOCAL DIRECTORY '/tmp/local_out'
        SELECT a.* FROM pokes a;

hive> INSERT OVERWRITE TABLE events
        SELECT a.bar, COUNT(*) FROM invites a
        WHERE a.foo > 0 GROUP BY a.bar;
hive> SELECT t1.bar, t1.foo, t2.foo
        FROM pokes t1 JOIN invites t2
          ON t1.bar = t2.bar;
```


# MapReduce 程序设计进阶

## 高级特性支持

复合键值对: 

  - 将value的一部分移入key，并重写Partitioner，以节约部分存储、并实现自动排序
  - 可以考虑map不是发射一组键值对，而是将其编码为为一个HashMap的字符串表示，可以节约数据传送

用户自定义数据类型: 实现WritableComparable接口

  - write(DataOutput out)
  - readFields(DataInput in)
  - compareTo(T other)

用户自定义输入输出格式: 用于解析输入分片、格式化输出结果

用户自定义Partitioner和Combiner: 

  - 定制Partitioner以改变Map中间结果发送到哪个Reduce节点的分区方式
  - 定制Combiner以在Map端局部reduce、减少数据发送量（注: 不一定被调用，**不可控**

迭代式MapReduce:  如PageRank，反复在同一组输入输出文件上执行同一个MR Job  
组合式MapReduce: 多个MR Job按依赖关系组成DAG，然后JobContro会根据拓扑排序的顺序执行  
链式前处理和后处理: ChainMapper、ChainReducer可以让一个Job有多个map和reduce节  

多数据源连接: 

  - 用DataJoin类实现Reduce端Join: 数据量大的情况
  - 用DistributedCache机制、直接复制文件实现Map端Join: 数据量小的情况

全局参数传递: Configuration对象，用起来像个字典 get/setDateType()  
全局数据文件的传递: 

  - Job类中: job.addCacheFile(URIuri)将一个文件放到DistributedCache中
  - Mapper/Reducer的context对象中: context.getLocalCacheFiles() 获取DistributedCache中的文件


划分多个输出文件集合: reduce之前会决定这一组数据写入哪个文件

```java
public static class SaveByCountryOutputFormat extends MultipleTextOutputFormat<NullWritable, Text> {
  protected String generateFileNameForKeyValue(NullWritable key, Text value, String filename) {
    String[] arr = value.toString().split(",", -1);
    String country = arr[4].substring(1,3);
    return country + "/" + filename;
  }
}

job.setOutputFormat(SaveByCountryOutputFormat.class);
```

## PageRank

PageRank 是一种在搜索引擎中根据网页之间相互的链接关系计算网页排名的技术，PR值越高说明该网页越受欢迎  

基本设计思想: 被许多优质网页所链接的网页，多半也是优质网页  
获得较高PR值条件: 有很多网页链接到它 or 有高质量的网页链接到它  

计算RP的模型有两个，**图论简化模型**: 

```
互联网上的各个网页之间的链接关系看成一个有向图，对于任意网页 Pi 的PR值为

R(Pi) = Σ[Pj∈Bi](R(Pj) / Lj)

  Bi   所有指向Pi的网页    (in)
  Lj   Pi所指向的所有网页  (out)

定义超链矩阵H: H[i][j] = if Pj ∈ Bi then 1/Lj else 0
           R = [R(Pi)]
则易得 R = HR，即 R 是 H 对应特征值1的特征向量

该模型面临的问题: 
  Rank Leak: 没有出链的网页会使得所有节点的PR逐渐丢失至0
  Rank Sink: 没有入链的网页自己的RP会一步丢失为0
```

或者 **随机浏览模型**: 

```
假定一个上网者从一个随机的网页开始浏览
上网者不断点击当前网页中的任意链接开始下一次浏览
但是，上网者最终厌倦了，开始了一个随机的网页
随机上网者用以上方式访问一个新网页的概率就等于这个网页的 PR 值
*这种随机模型更加接近于用户的浏览行为

矩阵表示:  H' = d*H + (1-d)*[1/N]_(N×N)
           R = H'R
其中: 
    H' 代表转移矩阵
      d      阻尼因子(通常d=0.85), 即按照超链进行浏览的概率
      1-d    随机跳转一个新网页的概率
    R 为列向量，代表 PageRank 值
*由于等式 R=HR 满足马尔可夫链的性质，如果马尔可夫链收敛，则 R 存在唯一解
```

MapReduce实现PageRank，使用随机浏览模型: 

  1. GraphBuilder，建立网页之间的超链接图
    - map: <*, html> -> <\url, (initPR, outLinks)>     // 解析每个html的出链，并随机初始化initPR
    - reduce: <*, *>                                   // 不做任何事
  2. PageRankIter，迭代计算各个网页的 PageRank 值
    - map: <\url, (curPR, outLinks)> -> [<\u, curPR/len(outLinks)> for u in outLinks] and <\url, outLinks>
    - reduce: <\url, (newPR, outLinks)>
    - *终止条件: PR收敛、排名收敛、固定迭代次数
  3. RankViewer，按PR值从大到小输出
    - map: <\url, (curRP, outLinks)> -> <\url, curRP>  // 提取最后一次的newPR作为curRP
    - reduce: <*, *>

## K-Means

并行化K-Means思路: 

 - 将所有的数据分布到不同的节点上，每个节点只对自己的数据进行计算
 - 每个 Map 节点能够读取**上一次迭代**生成的 cluster centers，以判断自己的各个数据点应该属于哪一个
 - 每个 Reduce 节点综合每个属于每个 cluster 的数据点，计算出新的 cluster centers

需要全局共享的数据: 

  - 当前的迭代计数
  - K 个表示不同聚类中心的数据结构: (cluster_id, cluster_center_point, cluster_size)

MapReduce程序流程(迭代多次直到收敛/指定次数): 

  - setup: 读取共享文件，获得 centers 数组
  - map: <*, point> -> <\cluster_id, (p, 1)>                     // 把p放到其最近中心的簇里
  - combine: <\cluster_id, [(p, 1)]> -> <\cluster_id, (p', n)>   // n = len([(p, 1)]), p' = avg([p])拟将是新的簇中心
  - reduce: <\cluster_id, [(p', n)]> -> <\cluster_id, (pm, n')>  // pm是新的簇中心，写入到全局共享的数据中

## 分类

### kNN

MapReduce程序流程: 

  - idea: 简单的分块分治，计算每个测试数据的kNN的分类、取最多投票
  - job: 将训练数据文件放在DistributedCache中供每个节点共享访问
  - map: <*, (X, y)> -> <\X, y'>      // 对每个测试数据计算其kNN，并确定预测值y'
  - reduce: <*, *>

### Naive-Bayes

MapReduce程序流程: 

  - idea: 分治、滤波器模型、两阶段串联，计算每个测试数据落入每个分类的概率、取最大者
  - map: <*, (X, y)> -> <\y, 1> and [<(y, x.name, x.val), 1)> for x in X]
  - reduce: 
    + <\y, 1> -> <\y, n>                                   // 累加得到分类的频数
    + <(y, x.name, x.val), 1)> -> <(y, x.name, x.val), n>  // 累加得到属性频数
  - job: 将上一步所得频数数据放在DistributedCache中供每个节点共享访问
  - map: <*, (X, y)> -> <\X, y'>    // 对每条测试数据计算其每个滤波器的概率，以最大者作为预测值y'
  - reduce: <*, *>

### SVM

MapReduce程序流程: 

  - idea: 针对多分类问题，并行地为每个分类训练一个二分类器
  - map: <*, (X, y)> -> <\y, (true/false, X)>       // 对每条训练数据，产生特征向量
  - reduce: <\y, [(true/false, X)]> -> <\y, clf>    // 训练模型
  - setup: 所有的clf模型放入DistributedCache
  - map: <*, (id, X)> -> [<\id, (y, score)> for x in X] // 对每条测试数据，在每个clf模型上产生预测结果
  - reduce: <\id, [(y, score)]> -> <\id, y>         // 选择得分最大者

## Apriori & SON & PSON

先验规则: 全局频繁必然局部频繁、局部非平凡必然全局非频繁  
基本思想: 设置候选项集从1-项集开始、删去非频繁项集，余下的两两交叉取并集得到新的候选项集，重复直到候选项集为空，择上一轮中的结果即为极大频繁项集  


# Spark

Scala语言特性: 

  - var/val: 可变/不变引用
  - class/object: 类/单例类
  - 方法与函数
    - 方法: `def add(x: Int): Int = x + 1`，这是语法结构
    - 函数: `val add = (x: Int) => x + 1`，这是个对象
  - 下划线符号作为通配符，常用于取代匿名函数的形参

MapReduce的缺陷: 

  - 设计为高吞吐批处理，因而高延迟、不实时
  - Jobs之间数据共享要经过HDFS I/O，对迭代计算不友好
  - 内存使用不佳，JVM开进程消耗大量系统资源
  - MR计算模型对图论计算、迭代计算的表达力不强

Spark的口号: Speed速度、Ease of Use易用性、Generality广泛性、Runs Everywhere多处运行

Spark的基本构架和技术特点:
  
  - 节点拓扑: 主从模式，一个主节点MasterNode，和若干个工作节点WorkerNode(可以有多组Worker + Executor)
  - 基于内存计算的弹性分布式数据集(RDD): Transformation/Action实现惰性计算
  - 灵活的计算流图(DAG): 记录RDD转变的世系关系，当RDD丢失时递归重计算父RDD，容错和鲁棒性更好
  - 多种计算模式: 查询分析，批处理，流式计算，迭代计算，图计算，内存计算
  - 事件驱动的调度方式: 采用事件驱动的Scala库类Akka来完成任务的启动，复用线程池以取代MapReduce进程或者线程开销
  - *综上，是一种基于内存的迭代式分布式计算框架

Spark的数据抽象: 弹性分布式数据集 Resilient Distributed Dataset (RDD)

  - 能横跨集群所有节点进行并行计算的分区元素集合
  - 可从 HDFS 中的文件中创建而来，或者从一个已有的 Scala 集合转换得到
  - 使用对应的 Transform/Action 等操作算子执行分布式计算
  - 基于 RDD 之间的依赖关系组成 **世系**(即计算谱图) + 重计算, 检查点 等机制来保证容错性
  - 只读、可分区，全部或部分可以缓存在内存中、以多次重用
  - 弹性是指内存不够时可以自动与磁盘进行交换

RDD的容错实现: 

  - Lineage 世系/依赖: Lineage记录了计算图，而不需要存储太多实际的数据，就可以通过重计算完成数据的恢复，使得 Spark 具有高效的容错性
  - CheckPoint 检查点: 对于 lineage 很长的 RDD 来说、通过 lineage 恢复耗时较长，因此在对包含宽依赖的长世系的 RDD 设置检查点操作非常有必要 (RDD的只读性使得checkpoint容易完成)

RDD依赖关系: 

  - 窄依赖: 父 RDD 中的一个 Partition 最多被子 RDD 中的一个 Partition 所依赖
  - 宽依赖: 父 RDD 中的一个 Partition 被子 RDD 中的多个 Partition 所依赖，比如有join/union

RDD持久化: 

  - 未序列化的 Java 对象，存于内存中
  - 序列化的数据，存于内存中 (适度压缩)
  - 磁盘存储

Spark语法入门例子: 

```java
lines = spark.textFile("hdfs://..")           // Base RDD
errors = lines.filter(lambda s: s.startswith("ERROR"))  // Transform
messages = errors.map(lambda s: s.split('\t'))[2])      // Transform
messages.cache                                // cache it!
messages.filter(lambda s: "foo" in s.count)   // Action
messages.filter(lambda s: "bar" in s.count)   // Action
```

Spark程序结构和基本概念: 

  - 主节点: 
    - Application = 1 * Driver Program + n * Executor
    - Driver Program: 执行用户代码的 main() 函数，并创建SparkContext
    - Cluster Manager: 集群当中的资源调度服务选取，如standalone manager, Mesos, YARN
    - Job: 由某个 RDD 的 多个Transformation算子 + 一个Action算子 生成或者提交的调度阶段，每执行一次Action操作就会提交一个Job
    - SparkContext: spark集群逻辑上在用户程序中作为一个单例对象存在，它负责与主节点通信
      - 创建 SparkConf 类的实例
      - 创建 SparkEnv 类的实例
      - 创建 TaskScheduler 和 DAGScheduler 类的实例
  - 从节点: 
    - Executor: 负责在子节点上执行 Spark 任务
    - Stage/Taskset = n * Task，每个Action操作产生一个Final Stage、每个Shuffle操作会产生一个Shuffle Stage (Shuffle只在宽依赖时才发生)
    - Task: 基本执行单元、在一个Executor上完成；作用的单位是Partition，针对同一个Stage会分发到不同的 Partition 上执行 (??)

Spark程序的执行过程: 

  1. 用户编写 的 Spark 程序提交到相应的 Spark 运行框架中
  2. Spark 创建 SparkContext 作为本次程序的运行环境
  3. SparkContext 连接相应的集群配置 Mesos/YARN 来确定程序的资源配置使用情况
  4. 连接集群资源成功后， Spark 获取当前集群上存在 Executor 的节点，准备运行程序并且确定数据存储
  5. Spark分发程序代码到各个节点
  6. SparkContext 发送 Tasks 到各个运行节点来执行

Spark编程示例: 

```java
def main( args : Array[String]) {
  val conf = new SparkConf.setAppName ("Spark Pi")
  val sc = new SparkContext(conf)
  val fileRDD = sc.textFile("hdfs://root/Log") // RDD[String]
  val filterRDD = fileRDD.filter(line => line.contains("ERROR"))
  result = filterRDD.count
  println(result)
  sc.stop
}

[RDD的创建]
sc.textFile("...")           // 从HDFS
sc.parallelize(1 to 100, 2)  // scala集合
rdd.file.filter(...)         // 从现有rdd上Transform

[RDD的操作: Transform/Action]
rdd.map
   .flatMap
   .filter
   .sample
   .union
   .intersection
   .distinct
   .join
   .cartesian
   .groupByKey
   .reduceByKey
   .aggregateByKey
   .sortByKey
rdd.reduce
   .foreach
   .collect
   .count
   .countByKey        // collect(key).distinct.count
   .first             // take(1)
   .take
   .takeSample        // sample(n).take(n)
   .saveAsTextFile/saveAsSequenceFile/saveAsObjectFile
```

## wordcount

```js
val file = spark.textFile("hdfs://in")
val counts = file.flatMap(line => line.split(" "))
                 .map(word => (word, 1))
                 .reduceByKey(_ + _)
counts.saveAsTextFile("hdfs://out")
```

## k-means

```js
import spark.util.Vector

val lines = sc.textFile("...")
val data = lines.map(s => s.split(" ").map(_.toDouble)).cache
val kPoints = data.takeSample(false, K, 42).map(s => Vector(s))

for (i <- 0 until 100) {
  var closest = data.map(p => (closestPoint(Vector(p), kPoints), (p, 1)))
  var pointStats = closest.reduceByKey { case ((x1, y1), (x2, y2)) => (x1 + x2, y1 + y2) }
  var newPoints = pointStats.map { pair => (pair._1, pair._2._1 / pair._2._2).collectAsMap }
}

println(newPoints)
```

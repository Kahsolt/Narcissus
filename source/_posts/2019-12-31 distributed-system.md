---
title: 分布式系统 - 笔记
date: 2019-12-31 11:07:46
updated:
categories: 神谕机
tags:
  - 分布式
---

<br/>

# 前言

这是**分布式系统**的课程笔记，推荐读物是: 

  - Anthony, Richard John. System programming: design and developing distributed applications.
  - Andrew S. Tanenbaum, Maarten van Steen. Distributed Systems: Principles and Paradigms (2nd edition).
  - George Coulouris. Distributed Systems: Concepts and Design (4th edition).

分布式系统: 一组自治计算单元向用户呈现为单一凝聚的系统  
分布式优点: 向外扩展的经济性、并行化速度优势、应用需求内在分布式、可靠性、增量扩展

分布式系统目标ATOS: 

  - 资源可用性Availability
  - 分布透明性Transparency
    - 访问透明: 隐藏数据表示差异，以及资源访问方式
    - 位置透明: 隐藏资源所在位置
    - 迁移透明: 资源发生位置迁移
    - 重定位透明: 正在使用中的资源发生位置迁移
    - 复制透明: 资源有多个副本备份
    - 并发透明: 资源被多个用户竞争式共享
    - 错误透明: 隐藏错误处理
      - 通常不能完全透明: 无法区分一个慢节点和宕机节点、无法确信命令在宕机前得到了执行
  - 开放性Openness: 至少做到独立于异构环境
    - 良定义的接口，以支持系统间交互操作
    - 应用可移植性
  - 可扩展性Scalability
    - 数量可扩展: 用户数、进程数
    - 地理可扩展: 节点间最大距离
    - 管理可扩展: 管理域数量

分布式系统分类: 

  - 分布式计算系统
    - 集群计算Cluster: 同构节点、高端软硬系统、局域网连接、单一管理节点
    - 网格计算Grid: 易购节点、散布多个组织、广域网
    - 云计算Cloud: IaaS(数据中心/AmazonEC2)、PaaS(MS-Azure/AmazonS2)、SaaS(GoogleApps/YouTube/Flickr)
  - 分布式信息系统
    - 事务处理系统(A原子/C一致/I隔离/D持久化)
  - 分布式普适系统
    - 普适计算系统
    - 移动计算系统
    - 传感器网络


# 架构

架构风格: 

  - 中心化架构
    - 层次化风格: Client-Server系统 (request-response flow)
      - 单Server系统问题: Server成为瓶颈、单点故障源，扩展难
      - 传统三层模型: 用户接口层、处理层、数据层
      - 瘦客户端: 表示层放在用户侧，例如X11server、移动应用
    - 基于对象的风格: 分布式对象系统 (object-method_call)
    - 发布订阅模式: 空间解耦/匿名化 (publisher-event_bus-notification_delivery-subscriber)
    - 共享数据空间: 时间空间解耦/异步化 (publisher-public_data_storage-subscriber)
  - 去中心化架构
    - 结构化P2P: 节点按某个规范的数据结构来组织，如逻辑环、超立方，找数据靠哈希
    - 无结构P2P: 节点邻居是随机选择的，找数据需要泛洪、随机游走
    - 混合P2P: 一些超级对等体(superpeer)节点被指派了特殊功能，存储数据索引、监视网络状态、为新入节点提供setup配置
  - 混合架构
    - C-S混合P2P
    - Edge-Server结构，用于内容分发网络CDN (Content Delivery Network)
    - BitTorrent: 下载同一个文件的用户组成一个临时小群
    - 中间件Middleware: 用于协调适配，自配置、自管理、自恢复、自优化(自主计算)，反馈控制模型


# 进程

进程: 在进程状态上下文中的一个执行流

  - 执行流: 指令执行流(步骤)、运行着的代码片段、连续的指令序列、线程的控制
  - 进程状态: 格局/快照、任何运行代码能影响和受影响的东西

进程与程序: 一个进程基本上是一组程序的运行时  
进程状态: 创建、等待、运行、阻塞、销毁  
进程低效原因: 创建进程时资源拷贝、调度时上下文切换、交互操作IPC等待/加解锁  

线程: 最小调度单位和局部上下文资源

上下文切换: 

  - 处理器上下文: 寄存器文件(PC/SP等)
  - 线程上下文: 处理器上下文 + 局部代码块状态(少量内存值)
  - 进程上下文: 线程上下文 + 线程状态(含MMU寄存器值)
  - *线程共享同一个地址空间

用户级线程: 

  1. 在用户进程地址空间被创建
  2. 优点: 一个进程覆盖所有操作，实现数据共享简单
  2. 缺点: 缺乏操作系统支持，难以灵活阻塞

系统级线程: 

  1. 作为系统调用使用，效率有所下降
  2. 不存在阻塞问题，处理外部事件简单

分布式系统中线程的应用: 多线程并行RPC、隐藏网络延迟、提高性能、更优的结构

虚拟化: 进程虚拟机(JVM)，虚拟机监视器(VMware/KVM/XEN)

服务器: 

  - 超级服务器: 监听多个端口，提供多项服务
  - 迭代/并行处理服务器: 一个处理一个请求/可同时处理多个请求
  - 有状态的/无状态的: 是否记录客户端访问、以及客户端个体识别

代码迁移: 迁移代码段+数据段、然后重启计算，或迁移整个对象状态、从断点继续计算


# 通信

通信方式分类: 

  - 易失通信: 发送/转发失败则丢弃，如UDP
  - 持久化通信: 发送/转发失败则暂存重试，如mail
  - 同步通信: 请求提交时同步/请求发送时同步/请求处理后同步，如TCP
  - 异步通信: 中断机制
    - 单向通信: client甚至不用等server的接受响应(类似UDP)

基本RPC操作: 

  - Client过程调用client存根(stub)，存根构造消息、调用本地OS接口
  - 本地OS发送消息给远程OS
  - 远程OS收取消息给远程存根，远程存根解包消息、根据参数调用服务
  - 服务进行本地调用并返回结果，远程存根构造消息、调用远程OS接口
  - 远程OS回复消息给本地OS
  - 本地OS收取消息给存根，存根解包消息、返回结果给Client

RPC错误: 

  - client找不到server/server挂了/stub版本不匹配: 特殊返回值(如-1)、抛异常、发信号
  - request丢失: 内核开计时器超时重发
  - reply丢失: 内核开计时器超时重发、请求附加序列号防止多次执行
  - server收到request后崩溃: 等server重启后重发请求(至少一次执行)、直接放弃报错(至多一次执行)、随机行为(啥也不担保)
  - client收到reply后崩溃: client发RPC之前记录日志以便崩溃重启后通知server清理孤儿、日志也可改用任期号来分代、server限制任务保留时间超时kill

client如何找server: 

  - 硬编码写死
  - 动态绑定: 需要有一个binder
    - register: (name,version,handle,uuid) -> null
    - deregister: (name,version,uuid) -> null
    - lookup: (name,version) -> (handle,uuid)

基于消息的通信: 

  - 抽象原语: send/receive
    - 属性: synchronous/asynchronous, persistent/transient
    - 易失消息通信选项: 易失异步通信、基于收据的易失同步通信(记下日志就返回)、基于交付的易失同步通信(开始处理就返回)、基于回复的易失同步通信(处理完才返回)
    - 持久消息通信选项: 持久异步通信、持久同步通信
  - 实例: TCP/IP Socket
    - 原语: socket,bind,listen,accept,connect,send,receive,close (下面|p|表示同步点)
    - server: socket -> bind -> listen -> |accept| -> |(read/write)*| -> close
    - client: socket ------------------> |connect| -> |(write/read)*| -> close

基于消息的中间件原语: PUT,GET,POLL,NOTIFY

面向流的通信: 

  - 数据值是时间依赖的: 多媒体数据、传感器采样数据
    - 流数据: 等时性、单向性
  - QoS: 比特率、最大会话启动延迟(setup)、最大端到端延迟(transfer)、最大抖动延迟(类似方差，用buffer解决)、最大往返延迟(round-trip)、丢包率(用交错分帧解决)、流同步(multiplex流整合后交给设备multiplex)
  - 多播: 
    - Application-level multicasting
    - Gossip-based data dissemination
      - Anti-entropy: each replica sync with exactly one another randomly chosen, O(logn)
        - 更新方法: push(p发更新给q)/pull(q去拉p的更新)/push-pull(p和q互换更新)
      - Gossip: a replca inform a number of k others of its update changes, with probability 1/k to stop


# 命名

可命名实体: 

  - 命名方式
    - 人类友好名字: 易不经意重名
    - 地址: 可能会移动，一个实体可能有多个地址
    - 标识符: 标识符最多指向一个实体、实体最多被一个标识符所指向
  - 名字解析
  - 名字空间

根据名字找访问点: 

  - 简单方法: 广播(不超过局域网)、前递指针(留存引用)、基于家的方法(代理转发，如蜂窝网络)
  - 分布式哈希表: 节点形成逻辑环，每个节点用m位命名、每个资源用m位命名、名字为k的资源落在最小的节点id>=k上，然后在环上顺着找
  - 分层位置服务: 考虑IPv6、SNMP协议

命名空间: 参考DNS协议、Linux文件系统


# 同步

同步纵览: 

  - 时钟: 物理时钟、逻辑时钟
  - 共享同步: 互斥锁
  - 选举: 决定分布式系统的协调者

目前的原子时: 来自世界50多个实验室的铯原子钟报时的统计平均

关系"发生在之前": 反自反、反对称、传递  
Lamport算法: 每个进程Pi拥有一个计时器Ci

  1. 对于Pi上接续发生的两个事件，Ci加一
  2. Pi所发送的每个消息m都带上时间戳`ts(m) = Ci`
  3. 当Pj收到任何消息m之后，总是调节自己的时钟为`max{Cj, ts(m)}`

并发条件下保证事件发生顺序: 用消息队列中间件去做排序

分布式互斥锁: 

  - 仲裁: 有一个仲裁者去决定是否可以进临界区(自己就是个mutex)  
  - 令牌: 令牌在环上轮转，谁拿到谁就可以进  
  - 投票: 
    1. 想要进入临界区的Pi构造消息`m = {critical_area_id,proc_id,cur_timestamp}`
    2. Pi广播m以征求意见，其他的Pj做如下操作: 
    3. 如果Pj不在临界区、也不想进临界区，回复同意
    4. 如果Pj在临界区，将m入队列
    5. 如果Pj不在临界区、但想进临界区，就比较时间戳，对方小则同意并自己入队，否则对方入队

分布式互斥锁对比: 

|  算法   |  消息数 |    时延   |    难题    |
|---------|--------|-----------|------------|
| 仲裁中心 | 3      | 2         |   单点故障  |
| 分布投票 | 2(n-1) | 2(n-1)    | 任何进程崩溃 |
| 令牌环   | 1 ~ ∞  | 0 ~ (n-1) | 进程崩溃令牌丢失 |

领导者选举: 

  - 欺负算法: 多播、比较谁编号大
  - 环算法: 发起者发一个问卷、每人轮流填，最后发起者检查谁应该称为领导人并通知它


# 一致与复制

为啥要做复制: 可靠性(容错)、性能(并行化)

一致性模型: 

  - 数据中心一致性: 
    - 严格一致性: 任何read(x)总返回最近的write(x)结果，任何write(x)立马被所有process察觉，**分布式系统无法实现**
    - 序列一致性: 事件是全序的；进程内、对同一个变量的**读写保持顺序**；进程间、看到的变量的**变更顺序**是一致的，**任何顺序执行结果都是相同的**，read不一定按时间顺序
    - 线性一致性: 进程内外看到对同一变量的操作都是严格地完全相同的顺序(进程串行?)，用于程序形式化验证
    - 因果一致性: 序列一致性的弱化，单进程的连续读写必须被其他进程按此顺序看到、但多进程间的并发写可以被其他进程看到不同的顺序(只要拓扑序存在即可?)
    - FIFO一致性: 单进程的**写**必须被其他进程看到一样的顺序，但多进程并发写可以被看到不同顺序
    - 弱一致性: 显式同步数据以后，共享数据才被认为是一致的
    - 释放一致性: 使用锁，先加锁后解锁、退出临界区时共享数据得到同步
    - 入口一致性: 使用锁，进入临界区时共享数据得到同步
  - 客户中心一致性: 
    - 最终一致性
    - 单调读: 读过一次之后，重复读不会得到更老的值
    - 单调写: 写是确定顺序的增量更新、不能跳着或逆序写
    - 写后读Read your write/读写一致性: 写过了，后续就会读到这个写的值
    - 读后写Write after read/写读一致性: ？

更新传播: 基于push/基于pull  

复制协议: 

  - Linearizability
    - Primary-Backup
    - Chain Replication
  - Sequential consistency
    - Primary-based
      - Remote-Write: 读任何节点、直接返回(2步)；写任何节点、转发到主、广播更新、承认更新、回复写确认(5步)
      - Local-Write: 读写时会把目前项目迁移到访问节点再处理
    - Replicated Write
      - Active replication
      - Quorum-based


# 容错

可信任性Dependability: 

  - 可用性: 系统达到了给定的权威规范、在给定时间内无故障的概率
  - 可靠性: 正常工作时间占比、在任何时点给出正常工作的概率
  - 安全性: 系统暂时失效时不会有灾难发生
  - 可维护性: 修复系统的难易度

关于错误: 

  - Failure: 系统规范说明里描述了的系统偏差
  - Erroneous State: 进入错误状态就会导致Failure而非Fault
  - Error: 系统总状态中出错/不期望的局部
  - Fault: 系统设计故障，错误原因
  - *Fault causes Error, Error results in Failure

增进可信任性: 信息冗余、时间冗余、软硬件冗余(三份冗余、进程组)

K-容错: 

  - 安静失效错误: 需要`K+1`个节点(至少一个存活)
  - 拜占庭错误: 需要`2K+1`个节点(从众选举)

拜占庭问题: 通信是完美的，但处理器不诚实

  - 口头协议: 有`m`个叛徒，则系统总共需要至少`3m+1`个节点
  - 书面协议: 有`m`个叛徒，则系统总共需要至少`m+2`个节点

分布式提交: 

  - 两阶段提交
    - cli给所有srv发选举请求，srv回应commit或abort
    - cli收集统计投票，若全部commit则发送global_commit给所有srv、否则发送global_abort，每个srv相应处理
  - 三阶段提交
    - pre_commit
    - ready_commit
    - global_commit

恢复: 利用checkpoint

  - 全部进程先恢复到最后一次保存点
  - 检查此间丢失的消息(发送撤回)
    - 若发送被撤回，则消息的接收者应该重新接受这些消息
    - 因此回退到对应的发送前保存点
  - 递归此过程
  - *只要所有已经记录了接收的信息都确实历史上发送过


# 专题*

## 云计算

云: 网络+资源池  
相关技术: 并行计算、分布式计算、网格计算  
云服务分类: IaaS,PaaS,SaaS  
虚拟化: 封装、隔离、多实例、硬件无关、特权(入侵/病毒检测)、资源动态调整  

流调度: 按包、按流；B4协议(区分背景流量与紧急流量，优先安排紧急、插空安排背景)  
带宽保证: 扩展频率带宽、网络结构优化、网络虚拟化  

云计算(Cloud Computing)

  - 网格计算(Grid Computing)
  - 分布式计算(Distributed Computing)
  - 并行计算(Parallel Computing)
  - 效用计算(Utility Computing)
  - 联机存储技术(Network Storage Technology)
  - 虚拟化(Virtualization)
  - 负载均衡(Load Balance)

OpenStack主要的服务成员:

 - 计算(Nova)
 - 存储(Swift对象/Cinder块)
 - 镜像(Glance)
 - 网络(Neutron)  

## 边缘计算

边缘计算: 利用靠近数据源的边缘地带计算资源来完成任务  
边缘设备: 各种嵌入式智能设备  
针对低端设备的的优化: 有效采样、优化带宽、仔细考虑缓存、多级算法(先用粗糙算法筛去一些数据)  

## 物联网

物联网四层模型: 

  - 感知识别层: RFID、无线传感器
  - 网络构建层: 3G/4G网络、WiFi、蓝牙、NFC
  - 管理服务层: 数据管理、Web服务、目录服务、权限管理
  - 综合应用层: 各种应用系统

RFID中的标签识别算法: 基于帧的分时隙ALOHA防冲突算法、基于二进制树的防冲突算法  
举了个例子: 摄像头识别纸键盘输入

---
title: 自然语言处理-笔记
date: 2019-12-25 08:59:46
updated: 2019-12-25
categories: 神谕机
tags:
  - NLP
---

<br/>

# 前言

这是**自然语言处理**的课程笔记，推荐读物是: 网课、其他外国大学的NLP讲义 (感觉我系NLP真的有些弱)

自然语言处理的基本实现方法流派: 

  - 理性方法 (基于规则/知识工程)
    - 基于字典和规则的形态还原、词性标注、分词
    - 基于CFG/扩展CFG的语法分析
    - 基于逻辑形式和格语法的语义分析
    - 基于规则的机器翻译
  - 统计方法 (基于统计/语料库)
    - 语言模型: n-gram
    - 分词、词性标注: 序列化标注
    - 语法分析: 概率上下文无关
    - 文本分类: 朴素贝叶斯、最大熵
    - 机器翻译: IBM模型
    - ...: 基于神经网络的深度学习方法


自然语言处理NLP的临近领域: 自然语言理解NLU(强调对语言含义和意图的深层理解)，计算语言学CL(强调可计算的语言理论)

自然语言处理的应用: 

  - 机器翻译MT/机器辅助翻译MAT: 文本、语音
  - 自动摘要: 单文档、多文档 (应对信息过载)
  - 文本分类: 网页、图书、信息过滤
  - 信息检索: 搜索引擎
  - 自动问答: QA机器人
  - 情感分析: 舆情分析、市场决策
  - 信息抽取: 实体、关系、事件等 (非结构/半结构化数据中抽取结构化信息)

自然语言分类(基于形态结构): 

  - 分析语: 没有/较少词形变化，没有表示词的语法功能的附加成分，由词序和虚词表示语法关系
  - 黏着语: 有词形变化(-~)，词的语法意义/功能由附加成分来表示
  - 屈折语: 有词形变化(-~-)，词的语法意义由形态变化来表示

自然语言处理的难点: 

  - 歧义
  - 语言知识的表示、获取和运用
  - 成语和惯用型
  - 对语言的灵活性和动态性的处理
    - 灵活性: 同一个意图的不同表达，甚至包含错误的语法
    - 动态性: 语言在不断的变化，新词
  - 上下文和常识(语言无关)的利用

汉语处理的难点: 

  - 缺乏计算语言学的句法/语义理论，大都借用基于西方语言理论
  - 语料库缺乏
  - 词法分析
    - 分词、词性标注难
  - 句法分析
    - 主动词识别难 (特别对于流水句)
    - 词法分类与句法功能对应差 (例如: 他喜欢走)
  - 语义分析
    - 句法结构与句义对应差 (例: 老头晒太阳)
    - 时体态确定难 (独立语)


# 基于规则的自然语言处理/理性方法/传统方法

## 词法分析

任务: 

  - 分词Tokenize: 识别出句子中的词
    - 消歧义
      - 交集型: ABC = A/BC or AB/C 
      - 组合型: AB = AB or A/B
      - 混合型: ABC = A/BC or AB/C or A/B/C (交集+组合)
      - 伪歧义: 根据自己的语义能确认如何最大熵切分，挨批评 = 挨/批评
      - 真歧义: 根据上下文的语义才能确认如何最大熵切分，从小学 = 从/小学 or 从小/学
    - 方法
      - 正向最大匹配(FMM)/逆向最大匹配(RMM)
      - 双向最大匹配(FMM+RMM): 冲突时按交集型歧义处理
      - 正向最大、逆向最小匹配: 冲突时按组合型歧义处理
      - 逐词遍历匹配: 每次去掉全句最长的词
      - 设立切分标记: 收集词首字和词尾字作为界符，适合日语或文言文
      - 全切分: 获得所有可能的切分，选择最大可能(比如最大熵)的切分
    - *难点: 语素vs单字词，词vs短语/词组 的界定
    - *问题: 丢失信息、分词错误、不同的分词规范
  - 形态还原Lemmatize: 把句子中的词还原成它们的基本词形
    - 根据语言学家给出的变形规则还原
    - 不规则情况特殊处理
  - 词性标注POS: 为句子中的词标上预定义类别集合中的类
    - 词类/品词
      - 开放类: Noun/Verb/Adjective/Adverb
      - 封闭类: Determiner/Pronoun/Preposition/Conjunction/Auxiliary Verb/Particle/Numeral
    - 查字典定词类，对于兼类词使用上下文消歧义
    - *难点: 英语中10.4%是兼类词
  - 命名实体识别NER: 识别出句子中的人名、地名、机构名等

## 语法分析

任务: 

  - 组块分析(浅层句法分析、部分句法分析): 基本短语(非递归的核心成分)识别
  - 组成分分析(结构分析，完全句法分析): 词如何构成短语、短语如何构成句子
    - 基于CFG表示，解析成AST
    - 算法: 递归下降、CYK/ChartParsing
  - 依存分析: 词之间的依赖/支配关系

基于特征的CFG: 

```
[regular-cfg]
S -> NP VP
NP -> ART N
NP -> ART ADJ N
VP -> V
VP -> V NP

朴素的CFG会产生不合语法的句子: 主谓不一致、不及物动词带宾语
*解决: 增加句法符号，或给句法符号带上一系列属性参数

[feature-based-cfg]
扩展后每个句法符号形如: 
NP(PER 3, NUM s)            // 第三人称单数
VP(PER 3, NUM p)            // 第三人称复数
NP(AGR (PER 3, NUM s))      // 特征值可以是一个特征结构，此例亦可或简写为 NP(AGR 3s)
N(ROOT fish, AGR {3s,3p})   // 特征值可以有多个
NP(AGR ?a)                  // 特征值可以是变量
NP(AGR ?a{3s,3p})           // 受限变量，约束变量取值范围
S -> NP(AGR ?a) VP(AGR ?a)  // 同名变量值相同，实现主谓一致约束

上例改进后的扩展文法为: 
S -> NP(AGR ?a) VP(AGR ?a)
NP(AGR ?a) -> ART N(AGR ?a)
NP(AGR ?a) -> ART ADJ N(AGR ?a)
VP(AGR ?a) -> V(AGR ?a, VAL itr)
VP(AGR ?a) -> V(AGR ?a, VAL tr) NP

[Unification Grammar]
一个文法可以表示成一系列特征结构间的约束关系所组成的集合，这样的文法称为合一文法(Unification Grammar, UG)，它为基于特征的CFG文法提供了形式描述基础
合一运算: 特征结构相容的并集(直觉上即递归地将特征键值对取并集，但交集部分的键值对**必须一致**)

(CAT V, AGR 3s) 与 (CAT V, AGR 3p) 不相容
(CAT V, ROOT cry) ∪ (CAT V, VFORM pres) = (CAT V, ROOT cry, VFORM pres)
(CAT N, ROOT fish, AGR {3s,3p}) ∪ (CAT N, AGR 3s) = (CAT N, ROOT fish, AGR 3s)
```

## 语义分析

任务: 

  - 词义计算: 词义表示、多义词消歧等
    - 词义表示
      - 义项/义位: 词典里的一条解释
      - 语义类: 泛化、近义义项归类，{走, 跑, 跳, 爬} -> 移动
      - 义素/义原组合: 义项再拆解，哥哥 -> {人, 亲属, 同胞, 年长, 男性}
    - 词义关系
      - 上下位/抽象-具体: 动物-狮子
      - 整体-部分: 身体-手
      - 同义: 美丽-漂亮
      - 反义: 高-矮
      - 包含: 兄弟-哥哥
      - 语义场: 师傅-徒弟，上-下-左-右
  - 句义计算: 逻辑形式与组合理论、语义角色标注等
    - 分类: 上下文无关意义/上下文有关意义
    - 分析方式
      - 先句法后语义
      - 句法语义一体化
      - 完全语义分析(无句法分析)
    - 句义表示
      - 逻辑形式LF: 对一阶谓词演算FOPC的扩展
      - 论旨角色或格角色
  - 篇章语义计算: 指代、实体关系等

语义的逻辑形式LF表示(近似Prolog): 

```
Fido is a dog.
(DOG1 FIDO1)
Sue loves jack.
(LOVES1 SUE1 JACK1)
Sue does not love jack.
(NOT (LOVES1 SUE1 JACK1))
Most dogs bark.
(MOST1 d1:(DOG1 d1)(BARKS1 d1))
John sees Fido.
(PRES(SEES1 JOHN1 FIDO1))

Every boy loves a dog.
(EVERY b1:(BOY1 b1)(A d1:(DOG1 d1)(LOVES b1 d1)))  // forall boy, exists dog, st. love(boy, dog)
(A d1:(DOG1 d1)(EVERY b1:(BOY1 b1) (LOVES b1 d1))) // exists dog, forall boy, st. love(boy, dog)
(LOVES1 <EVERY b1(BOY1 b1)> <A d1(DOG1 d1)>)       // 歧义表示
```

语义的论旨角色或格角色表示: 

```
以动词为中心，给出句子中其它成分与它的浅层语义关系
The boy opened the door with a key.
  the boy: AGENT 施事格
  the door: OBJECT 客体格
  a key: INSTUMENT 工具格

深层格的种类: 
  施事格(Agentive): **He** laughed.
  工具格(Instrumental): He cut the rope **with a knife**.
  与格(Dative): He gives me **a ball**.
  使成格(Factitive): John dreamed a dream **about Mary**.
  方位格(Locative): He is **in the house**.
  客体格(Objective): He bought **a book**.
  受益格(Benefective): He sang a song **for Mary**.
  源点格(Source): I bought a book **from Mary**.
  终点格(Goal): I sold a car **to Mary**.
  伴随格(Comitative): He sang a song **with Mary**.
  ...

动词格框架，词典中对每个动词需给出: 
  它所允许的格，包括它们的模态性质 (必需、禁止、自由)
  这些格的特征 (附属词、中心词语义信息等)

格语法分析
In the room, he broke a window with a hammer.
[BREAK
  [case-frame
    agentive: HE
    objective: WINDOW
    instrumental: HAMMER
    locative: ROOM]
  [modals
    time: past
    voice: active]]
```

## 机器翻译

基本策略: 

  - 直译(Direct): 从原文句子的表层(词、词组或短语)出发，直接转换成译文(必要的词序调整)
  - 转换(Transfer): 对源语言进行分析，得到一个**基于源语言的中间表示**；然后，把这个中间表示转换成基于目标语言的中间表示；从基于目标语言的中间表示生成目标语言
  - 中间语(Interlingua): 对源语言进行分析，得到一个独立于源语言和目标语言的、**基于概念的中间表示**；从这个中间表示生成目标语言

实现方法: 

  - 理性方法
    - Rule-based MT: 解析-推理-转换
      - 基于词: 字对字翻译、词序调整、形态顺变
      - 基于语法结构: 解析AST，树到树映射
      - 基于语义: 解析出谓词形式或格语法框架，然后作映射
      - 基于中间语言: 解析到一个基于概念的中间语言 (universal ontology是否存在?)
  - 经验方法
    - Statistical MT: 抽取一些统计特征
    - Example-based MT: 从大语料库出发、相似度计算找最近、局部魔改
    - Neural MT: 自动特征学习


# 基于统计的自然语言处理/经验方法

## Language Model

大多数语言都不是平坦的: 随意拼凑的一串字词大多数情况下都不在该语言中，如何计算一串词是一句话的概率  
P(W) = P(w1w2..wn) = ?   // 如何定义/计算此概率  

有限视野假设(Limited Horizon): 当前词出现的概率只和它前面的k个词相关 (k阶马尔可夫链/k-MC)  
*N-Gram模型 即 (N-1)阶马尔可夫链；适合意群密集、插入语少的句子

  - 2-gram/1-MC: P(W) = P(w1) * Πi[2,n]P(wi|w[i-1])
  - 3-gram/2-MC: P(W) = P(w1) * P(w2|w1) * Πi[3,n]P(wi|w[i-1]|w[i-2])
  - n-gram/(n-1)-MC: P(W) = P(w1) * P(w2|w1) * P(w3|w1w2) * ... * P(wn|w1w2..w[n-1])

模型参数: 

  - 假设词汇量为m，则2-gram参数量m^2、3-gram参数量m^3
  - 2-gram参数估计: `P(wi|w[i-1]) = P(w[i-1]wi) / P(w[i-1]) = Count(w[i-1]wi) / Count(w[i-1])`
  - 相对频率/最大似然估计 (??)

Zipf Law: 若以词频排序，词频与排位的乘积拟合一个常数

稀疏/零概率问题: 未观测到数据出现零概

  - 构造等价类
  - 参数平滑: 高概率者调低，小概率或者零概率者调高
    - Add Counts
      - P(wi|w[i-1]) = (Count(w[i-1]wi) + δ) / (Count(w[i-1]) + |V| * δ)    // |V|是所有可能不同的n-gram数量，对2-gram就是(总词汇量+1)
      - 分子分母同时加上一个较小的常数 (效果不好)
    - Linear Interpolation Smoothing
      - P(wi|w[i-2]w[i-1]) = λ1\*P(wi|w[i-2]w[i-1]) + λ2\*P(wi|w[i-1]) + λ3\*P(wi), λ1+λ2+λ3=1
      - 对于3-gram，带权平均其1-gram、2-gram和3-gram
    - Laplace Smoothing / Dirichlet Prior
      - P(wi|w[i-1]) = (Count(w[i-1]wi) + k*P(wi)) / (Count(w[i-1]) + k)
      - 是线性插值的特例，权不是常数而是函数
    - Katz Smoothing
    - Kneser-Ney Smoothing

数据集三分: 

  - 训练集 Training Data
    - 用来建立模型，获得模型参数
  - 测试集 Test Data
    - 从训练集以外独立采样
    - 反映系统面对真实世界的处理能力
  - 交叉确认集 Cross-Validation Data
    - 从训练集和测试集以外独立采样
    - 主要用来帮助做设计决策和参数设定(hyperparameters)

模型评估: 

  - 测试集: m个句子 `s1, s2, ..., sm`
  - 困惑度(Perplexity): 对测试集存在的概率评估，越小越好
  - `Perplexity = 2^(-l)`, 其中`l = logΠi[1,m](P(si))/m = Σi[1,m]logP(si)/m`、即所有测试集句子的预测成句概率之积取对数再除以测试集大小以求得平均

模型评价:

  - 成功之处
    - Speech Recognition
    - Optical Character Recognition, OCR
    - Context-sensitive Spelling Corretion
  - 不足
    - 依赖分词精度
    - 丢失远距离依赖的信息
    - 无结构化的、语法依赖信息

其他语言模型: 

  - n-gram: 依赖前面一个的窗口序列
  - cache: 依赖前面一个窗口集合 (可理解为无序n-gram)
  - grammar: 解析为语法树AST
  - NN-method: word2vec, sentence2vec, text2vec
    - skip-gram: 从给定词/词组预测上下文语境
    - CBOW: 从上下文预警推知词义不明的词
    - *处理长距离依赖(如从句插入): 使用RNN/LSTM

## Text Classification

朴素贝叶斯模型: 将文档D放入到最高后验概率的分类ck中

```
argmax[ck]P(ck|D) = argmax[ck]{P(D|ck)*P(ck)/P(D)} = argmax[ck]P(D|ck)P(ck)

如何表示文档D: 
  Bernoulli Document Model: 0-1向量/词袋、编码词典中每个词出现与否
  Multinomial Document Model: tf向量、编码词典中每个词出现的词频

如何计算 P(D|ck) 和 P(ck): 简单统计
  P(ck) = Count(ck) / N       // ck类的文档数 / 总文档数
  P(D|ck) = ???               // 所有ck类文档的向量求得一个平均文档ck-avg，计算D与ck-avg的相似度/归一化距离
```

模型评估: 

  - 它就是个线性模型，可以考虑ML的线性回归(这是个**分类**算法!)
  - 训练参数
    - likelihoods of each word given the class P(wt|ck)
    - prior probabilities P(ck)
  - 优化
    - 去除停用词: P("the"|ck) -> 1

文本特征:

  - Bag of words
  - Phrase-based
  - N-gram
  - Hypernym Representation
  - Using some lexicon or thesaurus
  - Graph-based Representation
  - Distributed Representation: word2vec, sen2vec, doc2vec……

特征选择: 

  - High dimensional space
  - Eliminating noise features from the representation increases eﬃciency and eﬀectiveness of text classiﬁcation 
  - Selecting a subset of relevant features for building robust learning models
  - Actually feature filtering
    - assign heuristic score to each feature f to filter out the *obviously* useless ones

Feature utility measures:

  - Stop words
  - Frequency – select the most frequent terms
  - Mutual information – select the terms with the highest mutual information (mutual information is also called information gain in this context)
  - Χ2 (Chi-square)


TF-IDF repretation for word:

  - f[i]j = frequency of term-i in document-j 
  - tf[i][j] = f[i][j] / max[i]{f[i][j]}    // normalize by dividing max frequency in the same document
  - df[i] = document frequency of term-i    // number of documents containing term-i
  - idf[i] = log2 (N / df[i])               // N is total number of documents
  - w[i][j] = tf[i][j] * idf[i]             // w[i] is a vector representing word-i

## POS & HMM

词性标注: 给某种语言的词标注上其所属的词类

  - 英例
    - The lead paint is unsafe.
    - The/Det lead/N paint/N is/V unsafe/Adj
  - 汉例
    - 他有较强的领导才能。
    - 他/代词 有/动词 较/副词 强/形容词 的/助词 领导/名词 才能/名词
  - *难点: 兼类词消歧(英语10.4%、汉语22.5%)

形式化为一个分类问题: 

  - X: (x1, x2, ...) 词串
  - Y: (y1, y2, ...) 词性串
  - 训练数据 (xi, yi)
  - 拟合函数 f: X -> Y

决定一个词词性的因素: 前后词的词性(基于词性的语法一定程度上确保了这一点)

```
[隐马尔科夫模型的直觉]
  名词  ->  动词 -> 名词 -> 名词
        x        x       x            // 随机过程状态转移链
 动名词 -> 动名词 -> 动词 -> 动词

    ?   ->   ?   ->   ?   ->   ?      // 隐状态
    |        |        |        |
   教授     喜欢      画       画      // 观测状态

[隐马尔科夫模型]
对于词串W = [w1]，词性串S = [si]，求P(S, W)？

s0 -> s1 -> s2 -> ... -> sn
      |     |      |     |
      w1    w2    ...    wn
P(S, W) = Πi P(wi|si) * P(si|s[i-1])   // 从词性串反推词串
```

马尔科夫模型: 

  - 马尔可夫过程
    - 一个系统有N个有限状态 S = {sn}
    - Q = [qt] 是一个随机变量序列，qt取值于S
    - 求 P(qt=sj|q[t-1]=si, q[t-2]=sk, ..., q1=sh)，即已知前t-1步所处状态，求第t步落在状态sj的概率
  - 两大假设
    - 有限视野: 只记得前t步，t是个常数
    - 时间独立性: 在状态si到sj之间迁移的概率是稳定的、不随时移

```
[马尔科夫模型示例: 天气预报]
模型: 1阶马尔科夫链
状态集合S: {雨, 多云, 晴}
初始分布Π: [ 0.33, 0.33, 0.33 ]   // 真实分布不明时，假装初始时是均匀分布
转移矩阵A: [ 0.4 0.3 0.3
            0.2 0.6 0.2
            0.1 0.1 0.8 ]
(根据S和A可画出对应的有限状态自动机DFA)
问题: 预测接下来的天气序列为 Q = 晴-雨-晴-雨-晴-多云-晴 的概率？
P(Q|Model) = P(S3,S1,S3,S1,S3,S2,S3)
           = P(S3|-)*P(S1|S3)*P(S3|S1)*P(S1|S3)*P(S3|S1)*P(S2|S3)*P(S3|S2)
           = Π3*A31*A13*A31*A13*A32*A23
           = ...
```

隐马尔科夫模型: 一阶马尔科夫模型的扩展

  - 状态序列满足一阶马尔科夫模型
  - 观测序列与状态序列之间存在一定**概率关系**
  - 输出独立性: `P(O1,...Ot|S1,..St) = ΠtP(Ot|St)`
  - **隐**: 状态及其转移不可见，可见的是观测序列

```
[隐马尔科夫模型形式化定义]
λ = (S, V, A, B, Π)
     S       状态集
       Q     状态序列，不可见 (∀q∈Q -> q∈S) 
     V       观测值集
       O     观测序列，可见 (∀o∈O -> o∈V)
     A       状态转移概率分布矩阵 A[i][j] = P(qt=sj|q[t-1]=si) // 第t-1步在si态，下一步转移到sj态
     B       观察值生成概率分布矩阵 B[i][k] = P(ot=vk|qt=si)   // 第t步在si态，产生观察值vk
     Π       初始状态概率分布向量 Π[i] = P(q1=si)

[词性标注HMM]
    S        预定义的词性标注集 pos
    V        文本中的词汇 token
    A        词性之间的转移概率
    B        某个词性产生某个词的概率 eg. P("I"|PRON), P("love"|ADV)
    Π        初始概率
```

Viterbi算法: 

  - 计算观察序列对应某一状态序列的概率
  - 动态规划思想，O(N^2*T)
  - 定义变量δt(i): 指在时间t时，HMM沿着某一条路径到达Si、并输出序列为w1w2..wt的最大概率
  - δt(i) = max P(...,qt=si, w1...wt) 

模型参数学习:

  - 给定状态集S和观察集V，学习模型参数A、B、Π
  - 模型参数学习过程就是构建模型的过程
  - 学习方法
    - 有监督: 最大似然估计
    - 无监督: Welch-Baum

其他模型对比: 

  - Naive Bayes
  - Logistic Regression
  - HMM: 生成式模型(Generative Model)
    - 计算联合概率 P(words, tags)
    - 除了生成tags，还生成words
    - 实际上，我们只需要预测tags
    - Probability of each slice = emission * transition = P(w[i]|tag[i]) * P(tag[i]|tag[i-1])
    - *很难结合更多的特征
  - HEMM: 判别式模型(Discriminative Model)
    - 直接计算条件概率 P(tags|words)
    - 预测tags，不需要考虑input的分布
    - Probability of each slice = P(tag[i]|tag[i-1], w[i]) or P(tag[i]|tag[i-1], [all words])
    - *容易引入各种特征
  - CRF
    - 是无向图，不再用概率模型去解释，而是用"团"/"势"来度量节点间的关系
    - 仍然是条件概率模型，不是对每个状态做归一化，而是对整个串/图做归一化，克服MEMM的标注偏置问题
  - Linear Chain CRF
  - Genral CRF

## Coreference Resolution

Eventually, what does that **pronoun refers** to?

Terms about 'refer':

  - referent: the entities or individuals **in the real world** that the text is pointing to
  - referring expression: the text that points to the referents
  - coreference: a set of referring expression strings that all refer to the same referent
  - singleton: if no coreference in the same text

Terms about similarity between to NP:

  - non-identity: not refer to the same referent in the context，even if they somewhat semantically similar without context
  - identity: they are almost certainly the same as far as one can tell
  - near-identity: one NP describes a subset of another NP (eg. a pronoun appears first, following many NPs describing each aspect of it)

Algorithms:

  - Hobbs algorithm (1978): 试着根据英语语法寻找所指代的NP
  - Stanford "Sieve": 一系列的模式匹配，从所有NP中筛去可能性低的

## Statistical Parsing

Ambiguities: PP Attachment

  - The children ate the cake with a spoon
  - PP: "with a spoon"
    - attach to "the children"
    - attach to "the cake"

Probabilistic/Stochastic context-free grammar (PCFG):

```
G = (T, N, S, R, P)
     T      set of terminals
     N      set of non-terminals
     S      the start symbol (one of the nonterminals)
     R      rules/productions of the form X -> γ
              X   a nonterminal 
              γ   sequence of terminals and nonterminals (possibly empty)
     P(R)   gives the probability of each rule

The grammar G generates a PCFG language model L = L(G)

[example]
w = "astronomers saw stars with ears"
PP = "with ears"
  - attach to "astronomers"
  - attach to "stars"
P(t) = 生成某棵AST t的概率，将每一步推导所使用规则的附加概率值相乘即可
P(w) = ΣjP(tj) = 句子w能产生的所有(含歧义)的AST tj的概率总和

构建两棵AST树，但由于PCFG的每一条文法规则都带一个概率值，故可以取P(t)概率较大的那棵树作为解析
```

How to learn the probability of each production rule:

  - maximum likelihood estimation
  - if we have a treebank: by law of large numbers, probability = relative frequence
  - if have not: expectation maximum estimation (EME) -- inside & outside probability

## Information Extraction

Named Entity Recognition (NER):

  - discover: People(PER), Orgnazition(ORG), Location(LOC), Geo-Political Entity(GPE), Facility(FAC), Vehicles(VEH), ...
  - BIO notation: B-cat(beginning), I-cat(inbody), O(other) (eg. Tim[B-PER] Cook[I-PER] is[O] from[O] Apple[B-ORG])
  - usually non-hierarchical, but nested is also frequent...
    - (√) [The University of California]ORG
    - (×) [The University of [California]GPE]ORG
    - (√) [[John]PER’s mother]PER

训练模型: MEMM, CRF, RNN, BiLSTM

## Machine Translation

机器翻译的发展: 

  - 机器翻译概念 [Weaver1949]
  - 规则机器翻译 [1950s]
    - 由语言学方面的专家进行规则的制订，一般包含词典和用法等组成部分
  - 实例机器翻译 [Nagao1980s]
  - 统计机器翻译 [Brown1993 Koehn2003 Chiang2005]
    - 从双语平行语料中自动进行翻译规则的学习和应用
      - 词对齐Word: 双语间同义语词符号的长度不一定相同、因而需要对齐
        - Easy一对一标记
        - Grid矩阵标记
      - 再辅以词同现，即可用诸如EM等方法训练翻译模型P(f|e)
    - SMT系统: 例从法语到英语
      - 从英法双语语料库和英语语料库独立训练一些统计特征形成模型A和B
      - 输入法语文本通过A变成破碎的英语文本  // Translation Model: P(f|e)
      - 再通过B变成英语文本                 // Language Model: P(e)
      - 有噪信道模型串联这两步              // Decoding Algorithm: argmax[e] P(f|e) * P(e)
  - 神经网络机器翻译 [Cho2014 Bahdanau2015]
    - 利用神经网络实现自然语言的映射
    - 利用递归神经网络实现源语言的编码和目标语言的解码
      - 优点: 适合处理变长线性序列 and 理论上能够利用无限长的历史信息
      - 缺点: 梯度消失 or 梯度爆炸 (解决: 长短时记忆LSTM)
    - 基于注意力的神经机器翻译
      - 利用注意力机制动态计算源语言端相关上下文

## 情感分析

相关近义术语: 情感分类(sentiment classification)、极性分类(polarity classification)、情感计算(affective computing)、倾向性分析、意见挖掘、情绪分析、立场分析、观点抽取、观点摘要、舆情分析、口碑分析  
广义情感分析: 立场分析(Stance classification)、情绪分类(Emotion classification)、情绪原因抽取(Emotion cause extraction)  
情感分析的特殊问题: 情感极性转移 and 情感领域适应  

情感分析任务粒度: 

  - 文档级: 需要概括处理
  - 句子级: 单一话题
  - 词语级: 形容词的褒贬、名词实物的评价 (eg. 面包 -> 褒, 真理 -> 贬)
  - 属性级: 被评论对象的属性 (eg. 外观 -> 好, 耐用性 -> 差, 续航 -> 一般)

文档/句子级情感分析:

  - 基于词典和规则的无监督
    - PMI-IR [Turney2002]
    - 通过对文本中所有短语的情感倾向性累加，根据正负判断文档的情感极性
  - 基于传统机器学习
    - [Pang2002]等等等
    - 特征工程: 位置信息, 词性信息, 词序及其组合信息, 高阶n元语法, 句法结构特征
    - 特征权重: TF, TF-IDF, 布尔权重
  - 基于卷积神经网络
    - [Kim2014] [Kalchbrenneret2014] [Zhang2015]
    - 数据: 词向量作卷积，得到抽象文本
  - 基于递归神经网络的情感分类方法
    - [Socheret2011-2013]
    - 数据: 句法树库treebank
  - 基于循环神经网络
    - [Shirani-Mehr2014] [Tai2015]
    - 数据: 词向量 + 注意力机制 -> 词权重
  - 基于层次编码
    - [Tang2015] [Yang2016]
    - 按照"词-句子-文档"层次化编码
  - 基于层次用户和商品注意力
    - [Wu2018]
    - 层次编码 + 用户注意力以及商品注意力机制

词语级情感分析:

  - 基于知识库
    - [Hu,Liu2004] [Blair-Goldensohnet2008]
    - 思想: 起手一个基本的褒贬种子词集，然后用同近义词、反义词词典、词距离来扩展
  - 基于语料库
    - [Hatzivassiloglouand-McKeown1997] [Wang,Xia2015]
    - 思想: 分析同现关系(相似模式出现的词情感相似)、联系关系(转折、递进、并列)
  - 基于表示学习
    - [Tang2014] [Vo,Zhang2016] [Wang,Xia2017]
    - 思想: 通过文档级的情感分析结果作差运算得到词的情感强度

属性级情感分析: 

  - 属性抽取方法
    - 基于传统机器学习
    - 基于深度学习

  - 属性情感分类方法
    - 基于词典和规则
    - 基于传统机器学习
    - 基于深度学习

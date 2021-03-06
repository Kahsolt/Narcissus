---
title: 可计算性入门
date: 2019-10-03 09:24:56
updated: 2019-10-03
categories: 砂时计
tags:
  - 计算理论
---

<br/>

# 从函数起步

早在小学时代我们就学会了加法，会在演草纸上化简形如`114 + 514 = ?`的算式以求得答案。
却要等到很久以后的某一天，才会恍然大悟——“原来加法运算就是个二元函数啊”——于是刷刷刷地写下一些相互等价的表述: 

  1. `f(x, y) = x + y`
  2. `Definition f (x, y : Nat) -> Nat := x + y.`
  3. `add/2`

数学老师教会我们如何用第一种写法高效处理成堆的模拟卷；
某些从太古代幸存下来的编程语言则在数据类型约束上十分严格，正如在第二种写法中，要求进一步指明输入的参数、返回的结果都是来自然数集`Nat`的元素；
第三种写法是最抽象而危险的，它仅仅告诉我们这个函数的名字叫add、它接受两个参数，我们只能根据这两条岌岌可危的信息去揣测更多细节。

让我们在这里回忆一下函数的三要素吧: 

  1. 定义域`D`，一个非空数集
  2. 值域`R`，另一个非空数集
  3. 法则`f`，从`D`到`R`的一个非空映射

由此确乎可见，写法二彰显了学究派锱铢必较的严格精神，而宛若草稿的第三种写法几乎完全没有给我们提供构成这个函数的任何一项有效信息——它只适合于速记——但我们仍然相信**它是一个函数**，可这又是什么意思呢？

{% blockquote %}
  函数是一个吃东西吐东西的黑箱子。
{% endblockquote %}

也就是说我们相信，只要给这个`add`喂两个东西，作为等价交换原则的忠实信徒，它就一定会吐出些什么结果来。
这是一个随处可见的比喻，其实在说: 非空数集上的映射即数的函变，函数的核心就在于那个决定把特定的某数变换成另一个特定的某数的法则。

我们对做任意的加法都很有自信，而不仅仅局限于`114 + 514 = ?`。
这里显然指的是，我们**能够**对任意的合法数字x和y，求得函数`add/2`的返回值——只因为我们深谙加法运算的法则——即使结果非常大、写下来要花很长的时间；但我们凭空确信，那也一定是**唯一确定的结果、有限可数的步骤、可计量的时间**。

——这个直觉式的信念被总结为这样一句话: **函数`add/2`是可计算的。**
（需要稍加注意的是，“可计算性”这个词根本上仅仅是用来描述**函数**的，它是函数的一个属性。）

难道还有不可计算的函数吗？
——很遗憾，有，大量的递归函数都是不可计算的，随手就是举一把栗子: 

  - f(x) = f(2 * x)
  - g(x) = if x == 1 then 1 else if x % 2 == 0 then 1 + g(x / 2) else 1 + g(3 * x - 1)
  - h(x) = if x == 1 then 1 else if x % 2 == 0 then 1 + h(x / 2) else 1 + h(3 * x + 1)

第一个例子中，除了`f(0) = 0`以外我们一无所知，在其他点上（比如x=2），计算式序列看起来“发散”了；
第二个例子是3N+1猜想的变种，函数g在某些点上可计算（比如x=3），在另一些点上计算式序列会产生循环圈（比如x=5）
第三个例子就是著名的**3N+1猜想**，它**看起来**对于任意x都可以有一个会终止的计算式序列，但我们目前却无法证明。

我们可能会开始疑惑了，这些鬼东西真的是函数吗？让我们检验一下三要素: 

  1. 定义域都是自然数集`N`
  2. 根据加法和乘法的封闭性可知，值域也应该是`N`的子集
  3. 法则都以递归的方式隐含了

仿佛没有太多问题，暂且让我们相信它们的确是函数吧 :)


# 可计算性

现在让我们试着形式化地定义**可计算性**的概念: 

**称函数f: D -> R是可计算的，当且仅当存在一个图灵机TM，使得对于任意x∈D，TM以x为输入，总在有限步骤内停机并打印出结果f(x)；同时也称该图灵机TM计算了函数f。**

**有限步骤内停机**的一个等价描述是: 把图灵机在计算过程中每一步的格局用一个关系连接起来称之为**计算历史**，则计算历史的长度是有限的，或者说计算历史是一个有穷集合。

从定义上就可以直接看出，可计算的函数和图灵机是一一对应的，且它们都等势于可数无穷集`N`（因为每个图灵机都可以编码为一个对应的有限长01串，而有限长01串的集合将之视作自然数集的二进制表示，也就是说等势于自然数集`N`）。

对于不可计算的函数，我们也能构造相应的图灵机去尝试计算它，但它不总在所有合法输入上停机，此时称它进入了“循环”——可能像上例中f(2)那样发散，也可能像g(5)那样产生一个状态循环。

## 不可计算的函数NC
这些病态函数的存在让我们稍稍警觉了起来，自然地，我们会有一个直观的想法: 如果不允许递归定义，是不是就可以把所有不可计算的函数都干掉了？
——很遗憾，不行。
让我们使用著名的对角线方法来构造一下这个著名的**不可计算函数NC (Not Computable)**。

| x\F | F_0 | F_1 | F_2 | F_3 | F_4 | F_5 | ... |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
|  0  |  *  |     |     |     |     |     |     |
|  1  |     |  *  |     |     |     |     |     |
|  2  |     |     |  *  |     |     |     |     |
|  3  |     |     |     |  *  |     |     |     |
|  4  |     |     |     |     |  *  |     |     |
|  5  |     |     |     |     |     |  *  |     |
| ... |     |     |     |     |     |     | ... |

如上表所示，在自然数集`N`上可定义的所有可计算函数F_i可以按照它们所对应的图灵机编号的大小排列出来，定义域`N`也是可列的，于是可以计算出对于任意`x∈N, i∈N`处的函数值`F_i(x)`，以填充满整个表格。
则存在一个函数NC，它的定义是对于任意`x∈N`，都有`NC(x) = F_x(x) + 1`（标星号的格子）。
由此，NC与所有的F_i都不相同，也即，它不是`N`上定义的可计算函数。

留意到**函数NC的定义不是递归的**。

一个可能的疑惑是，这不都有NC的定义了什么还不能计算，比如`NC(0) = F_0(0) + 1`，我们不是知道`F_0(0)`吗？
——这是两种意义上的知道，我们只知道所有可计算的函数可以列成一行，其中的第一个被命名为`F_0`、且在`x=0`处一定有定义也即有值；但我们并不知道`F_0`究竟是哪个函数、不知道它的具体定义，所以仍旧无从计算。


## 停机问题
考虑到函数定义是否是递归的并不是造成其不可计算病态的必要条件，我们又可以心安理得地使用递归函数了。这样一来，斐波那契数列的递推公式可以用了、错排公式也可以用了，生活又恢复了几分色彩。
——但是我们仍然还得适度提防那些不可计算的函数。

可是要怎么做呢，另一个想法应运而生: 我们能不能找到一种方法，有效判定一个函数是否可计算？
——很遗憾，也不行。
我们就遇到了图灵停机问题的变形。

先简化改写一下这个问题，用函数和图灵机两种等价表述方式: 

  - 是否存在一个接受一个函数和一个数为输入的可计算的高阶函数JUDGE(f : Function, x : Nat) -> Nat，定义为对于给定参数，如果`f(x) = 1`则返回1，否则返回0。（这种值域只有两个元素的函数叫做也谓词，对应的自然语言所描述的问题称为判定性问题。）
  - 是否存在一个图灵机TM，输入某个函数f和输入值x的字符串编码，如果`f(x) = 1`则打印1并停机，否则打印0并停机。（这种对于任意输入必定停机、且打印结果只有两种可能选项的图灵机也称为判定器/接受器，使得其打印1并停机的所有字符串组成一个语言，称该图灵机判定了该语言。）

我们从第一种即函数表述入手，假设存在这样的JUDGE，现在来构造另一个函数
  `REBEL(x) = 1 - JUDGE(F_x, x)`
注意到这里的`F_x`就是上面表格中编号为`x`的那个可计算函数。
简单地来说`REBEL`就是在`JUDGE`某个可计算函数`F_x`作用于自己的编号`x`之后的结果上再进行一次布尔取反；并且我们知道，既然`JUDGE`可计算，那么`REBEL`自然也是可计算的。那好，让我们尝试用REBEL计算自己的编号: 
```
  JUDGE(f, x) = if f(x) = 1 then 1 else 0 ; 根据定义
  REBEL(x) = 1 - JUDGE(F_x, x)            ; 根据定义

  (代入 x = _REBEL_，因为REBEL可计算的某个F_i，记其编号i为_REBEL_)
  REBEL(_REBEL_) = 1 - JUDGE(REBEL, _REBEL_)    ; F_[_REBEL_]就是REBEL函数自己
                 = 1 - (if REBEL(_REBEL_) = 1 then 1 else 0)
                 = if REBEL(_REBEL_) = 1 then 0 else 1

  因此，如果条件式中 REBEL(_REBEL_) = 1， 就会使得等式左右两边化简为 REBEL(_REBEL_) = 0
  反之同理，矛盾。
```
因此`REBEL`不可计算，`REBEL`的子函数`JUDGE`也不可计算。
就是是说，并不存在理想中能判定任意给定函数在某定义域上某处是否可计算的可计算函数`JUDGE`，就更别说判定任意给定函数在其定义域上各处都可计算的可计算函数了。


# 不必过敏

……那么，圆周率`π`可计算吗？
——可以！不要谨慎过头了w

由于我们只对函数使用可计算这个词，因此需要稍作变通，一个常数π可计算指的是这样一个特定函数`PI(n : Nat) -> Digit`可计算，即打印无理数`PI`在某进制表示下的第`n`位有效数字。

使用泰勒展开、欧拉公式、梅钦公式等等都可以有效计算任意给定数位上的数码，因为我们可以对于任意给定的`n`计算出用某级数逼近它只需要取前`F(n)`项即可达到误差范围。这个函数`F`根据所用的级数不同而有所不同，但都是确定且可计算的。
由此`π`是可计算的，几乎同理可知自然对数底数`e`也是可计算的。


虽然`NC`和`REBEL`敲碎了某些无谓的天真，但抛开刻意的人工构造而言，大多数物理问题形成的函数都是可计算的。
毕竟自然界可能比较喜欢简洁，因此不必过敏喵。


# 末后

许多问题都是相当开放的，答案是一个数字、几个公式、一篇自然语言短文都是可能的。
不妨让我们先考虑一些简单问题，也就是那些回答被限定在`Yes`或者`No`上的问题。
下一篇将集中讨论判定性问题，或者说谓词函数的可计算性，或者说语言判定器的存在性。
我们将遇到图灵机、可判定性、语言和文法之类的东西。


{% blockquote %}
  真正可怕的，还是直面无穷。
{% endblockquote %}

----

2019年10月3日 初稿

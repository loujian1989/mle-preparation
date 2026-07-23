1. Coding:
不是leetcode 但本质是Tree DFS.  不算难。但记得自己写code 测试自己的代码。
SD:
ad freq cap 经典了
DM:
demand side table。 经典了。就是advertiser/campaign /ad group/ ad 那套表
BQ1: 这个真没看出啥product think啥的， 和下面一轮我觉得都是BQ 但就是侧重点不一样吧
BQ2: 各种BQ 被问麻木了 经典ask /give feedback

# https://www.1point3acres.com/bbs/thread-1167571-1-1.html


2.电面拓扑
第二天给结果（一般两天内会给结果）。

昂塞
昂塞题的话和之前别人发的几乎一样，最近发的最常见的coding题目有拓扑、sd(freq cap)、data model，然后还有两轮BQ，分两天进行。

Coding 拓扑题
coding拓扑没什么好说的，不是原题，但本质一样。额外要注意的是记清楚 runtime，准备好一些变形，比如要 sort 多种结果或者打印全部结果以及相关的 runtime。

SD 题 （考验深度）
这里会挖得比较深，或者如果面试官没挖深，自己主动挖深一点，比如 Redis 如何实现，有几种模式，如何 handle traffic，如何做 atomic（为什么要做）。
大方向按 online/offline path 都讲一下（read/write）。offline 里用什么 MQ，怎么做 key，如果有特殊要求比如 strictly 1 ad cap 如何实现。

Data Model 题 （考验广度）
照着自己做过的或者 GPT 学的记一下，最好涵盖所有主要的广告部分，ad campaign、business account、creative、user targeting，能想到的都给他写上，具体聊下怎么联系的。（我也聊了下广告平台的整体流程，intake flow、serving flow、impression tracking flow，然后扩展到intake部分，这些可能没啥必要，只是展示下对整个系统的了解吧。）

BQ 两轮 （考察工作经验和jd是不是非常match，来了就能干）
都是老板聊，culture 没啥说的。常见的一些题我觉得这轮不会挂人。manager 轮基本是对你感兴趣的老板，如果你知道是哪个组或者你面的是哪个岗，把 job description 好好看下，
能保证自己做过 80%以上，然后聊经验的时候往上靠就行了。老板就是想要做过这些的人来了就干活。

基本上两天后就通知结果了，因为跟老板聊的很好就比较痛快，要做的东西就是我一直在做的所以面试的时候老板基本明示只要别的过了就直接来（干活）

一些我觉得比较重要的经验：
广告部门最近统招，有自己的题库和考量，我觉得基本属于半开卷了本身不难，如果没做过的部分用gpt多准备下就好，招人本质是找有相关经验的来了就干活，所以尽量保证面的岗和组和你自己的经验match80%以上比较容易过，老板轮应该是最关键的聊好了只要岗位在基本就没问题了
# https://www.1point3acres.com/bbs/thread-1167309-1-1.html


3.
Coding轮：
面试官一开始就说会分为三个部分：
第一部分很简单，不太记得细节了
第二部分是一道和力扣 伞 很像的用sliding window做的题目
第三部分：find the number of unique pairs of strings from a list such that the two strings in a pair share no common characters
Example:
input: {“apple”, “banana”, “peach”, “kiwi”}
unique pairs:  [apple, kiwi], [peaches, kiwi], [banana, kiwi]
面试官要求用O(nlogn) 的方法做, 我只想出来brute force的


Problem Solving 轮：讨论如何实现view port rendering on Netflix home page, 保证显示在首页的所有视频都没有重复的
我觉得这个Netflix blog还挺有帮助：https://netflixtechblog.com/lear...mepage-aa8ec670359a

# https://www.1point3acres.com/bbs/thread-1166298-1-1.html


4.
Technical Screen involving coding a cache. Interviewer asked several optimizations such as ensuring we don't run out of memory, how to handle scaling, etc. I answered all the questions well, 
# and discussed trade-offs. Got rejection email the next day.
# https://www.1point3acres.com/bbs/thread-1165026-1-1.html


5.
大哥上来自己generate 一个random number，得到6，于是问我第六题。

parse string to integer，不能用系统自带valueOf之类的

#1 可以有-，没有+，-永远在第一位，代表负数。如果overflow，throw exception。没有special character。
#2 可以有special character，字母之类的，只要有就throw exception
#3 可以有leading zero，只有一位leading zero是valid，超过一位leading zero就要throw exception。-可以在后来任何一位，这种情况要throw exception。

记不住了， 不是ads
# https://www.1point3acres.com/bbs/thread-1165013-1-1.html


6. （出新题了
1. coding
command undo system
func execute(command:str, tag:list[str])
func undo(tag:optional)-> str
如果没有给tag， undo最近的command， 如果有提供tag，undo带该tag的最近的command

2. SD
常规ads frequency cap 注意read/write path是分开的
3. data modeling on ads demand intake
这轮有点坑 我按常规顺序从account讲到impression 面试官都没提出什么不同意见 结果feedback说我没有dive into bidding, targeting etc...可是面试官也没有probe啊
4-5. 常规BQ
given and received feedback, pivoting(pivot fast), ownership, project with impact， conflict, ambiguity, how do you define success project outcome
两轮一直在问BQ 完全没有project deep dive 很疑惑 结果挂了告诉我我不够了解广告？？就一头问号 HM也完全没有问啊。。我猜他们希望看到candidate主动多分享细节吧

# https://www.1point3acres.com/bbs/thread-1164852-1-1.html


7.
Coding
经典拓扑排序题


SD
设计ads frequency cap系统


Data Modeling
常规的广告 entity schema, 广告组的同学应该都比较熟悉


BQ
都是常规问题，biggest mistake you made


Culture
How do you handle conflict and challenge

# https://www.1point3acres.com/bbs/thread-1164724-1-1.html



8.
recruiter screening 没啥特殊的 有提到Ads现在是centralized hiring 会在同一个pipeline里面试然后安排team match


店面 幺幺散留 followup是幺思酒思
三天后找recruiter followup 告诉过了 约VO


VO
SD - ads frequency cap
Data modeling - ads demand
Coding - 感觉像是面试官自己出的题 没啥难度 主要就是考数据结构的基本功
Culture， Product Thinking - 这两轮感觉差别不大 就是和Ads里面两个组的HM过简历然后BQ
说是2-3天会有结果 三天后followup recruiter完全不回我 以为凉了 结果又过了两天说有事耽搁了 好消息有个HM对我感兴趣 安排聊天


想问一下经历过的朋友 这个team match是不是说明VO里的两个HM都没看上我啊 后续match成功的概率高吗

补充内容 (2026-03-06 06:35 +08:00):
VO之后又和两个HM聊了天 断断续续等了一个月 今天recruiter打电话说要move forward with offer了


# https://www.1point3acres.com/bbs/thread-1164299-1-1.html



9.
Latency tracker class 提供两个功能：

1. 存latency samples。inputs：timestamp，latency in millisecond
2. 查询给定窗口内的，指定的percentile latency。input：time window，percentile。

该类需要支持concurreny。
# https://www.1point3acres.com/bbs/thread-1162820-1-1.html



10.
首先声明：本文是AI根据我写的草稿生成的

先说结论：我挂了。挂完的第一反应是“我是不是真的挂得有点委屈？”于是把整个过程写详细一些，代码、思路、沟通、踩坑，都摊开给大家看，欢迎拍砖。

---

## TPS（Technical Problem Solving）轮：题目澄清反复、时间不够用
- 时长：~60min
- 结果：挂
- 反馈：Need to collaborate more（多协作）

### 题目（我理解的版本在变化）
netlifx home page has a list of shelf，每个 shelf 有一堆 title。要在 viewport（每个 shelf 最多显示 X 个）里做dedupe。垂直滚动我们先不管（vertical viewport 视为无限）。

### 我的推进过程（v1 → v5）

#### v1：全局去重（拿到口头确认后开写）
当时我问了不少问题，面试官说 OK，我就开始写。结果写完发现他要的不是这个……

```python
def dedupe_v1(titles: list[list[int]], x: int):
  visited = set()
  ret = []
  for shelf in titles:
    current = []
    for title in shelf:
      if title not in visited:
        current.append(title)
        visited.add(title)
    ret.append(current)
  return ret
```

#### v2：只在前 X 个 unique 里全局去重（还是不对）

```python
def dedupe_v2(titles: list[list[int]], x: int):
  visited = set()
  ret = []
  for shelf in titles:
    current = []
    for title in shelf:
      if len(visited) <= x and title not in visited:
        current.append(title)
        visited.add(title)
      elif len(visited) > x:
        current.append(title)
    ret.append(current)
  return ret
```

#### v3：前 X 位跨 shelf 去重，之后只在各自 shelf 内去重（思路对，但实现有坑）

```python
def dedupe_v3(titles: list[list[int]], x: int):
  visited = set()
  ret = []
  for shelf in titles:
    current = []
    local = set()
    for title in shelf:
      if len(current) < x:
        if title not in visited:
          current.append(title)
          visited.add(title)  # 少了把前 X 位加入 local 的步骤
      else:
        if title not in local:
          current.append(title)
          local.add(title)
    ret.append(current)
  return ret
```

#### v4：最后 5 分钟Fix版（能跑但 if/else 很乱）

```python
def dedupe_v4(titles: list[list[int]], x: int):
  visited = set()

  ret = []
  for shelf in titles:
    current = []
    local = set()
    for title in shelf:
      dedupe = False
      if len(current) < x:
        if title not in visited:
          current.append(title)
          visited.add(title)
        else:
          dedupe = True        
      else:
        if title not in local:
          current.append(title)
      if not dedupe:
        local.add(title)
    ret.append(current)
  return ret
```

#### v5：面试后整理出的 clean 版本（结构清晰）

```python
def dedupe_v5(titles: list[list[int]], x: int):
  global_visited = set()
  ret = []
  for shelf in titles:
    current = []
    local_visited = set()
    for title in shelf:
      if len(current) < x and title not in global_visited:
        current.append(title)
        global_visited.add(title)
        local_visited.add(title)
      elif len(current) >= x and title not in local_visited:
        current.append(title)
        local_visited.add(title)
    ret.append(current)
  return ret
```

### 我的反思（TPS）
- 我确实有在问问题、拿确认，但还是三次理解偏差；说明我的提问方式和确认方式可以再优化（比如举具体 counterexample 让对方否定/肯定）。
- 时间被来回澄清吃掉，最后 5 分钟压出来的 v4 肯定不够 clean。
- “Need to collaborate more” 很可能指的是：更主动用例驱动对齐需求、边画边说、让对方看到你的思考分支，而不是“拿到一句口头确认就开写”。

---

## Coding 轮：第一题顺利，第二题在压力下写得不够简洁
- 时长：~60min
- 结果：挂
- 反馈：Code could be more concise/cleaner

### 问题 1：最大连续相同 show 的长度（15 分钟、一次过）

```python
def max_consecutive_streak(shows: list[int]) -> int:
  left = 0
  max_streak = 0
  for right in range(len(shows)):
    while shows[right] != shows[left]:
      left += 1
    max_streak = max(max_streak, right - left + 1)
  return max_streak
```

### 问题 2：最长连续不重复子数组（现场最终版过测，但不够“clean”）

先写了一个有 bug 的版本：

```python
def max_consecutive_unique_shows_with_bug(shows: list[int]) -> list[int]:
  left = 0
  ret = []
  visited = set()
  for right in range(len(shows)):
    visited.add(shows[right])
    while shows[right] in visited:
      visited.remove(shows[left])
      left += 1
    if right - left + 1 > len(ret):
      ret = shows[left:right + 1]
  return ret
```

当场我问“要不要手算还是直接线上 debug”，对方说都行；于是我线上 debug，定位到 bug（先 add 再检查）。

在时间压力下，我用了 dictionary 计数的思路，正确但不够简洁：

```python
def max_consecutive_unique_shows_with_final(shows: list[int]) -> list[int]:
    left = 0
    max_len = 0
    ret = []

    visited = dict()
    for right in range(len(shows)):
        cur = shows[right]
        while visited.get(cur, 0) > 0:
            visited[shows[left]] -= 1
            left += 1

        visited[shows[right]] = 1
        window_len = right - left + 1
        if window_len > max_len:
            max_len = window_len
            ret = shows[left:right + 1]

    return ret
```

更简洁的 set 解法（当时没想出来，但之后写了）：

```python
def max_consecutive_unique_shows(shows: list[int]) -> list[int]:
  left = 0
  ret = []
  visited = set()
  for right in range(len(shows)):
    while shows[right] in visited:
      visited.remove(shows[left])
      left += 1
    visited.add(shows[right])
    if right - left + 1 > len(ret):
      ret = shows[left:right + 1]
  return ret
```

最优的跳指针（post-interview by AI）：

```python
def max_consecutive_unique_shows_optimal(shows: list[int]) -> list[int]:
  left = 0
  ret = []
  visited = dict()
  for right in range(len(shows)):
    cur = shows[right]
    if cur in visited and visited[cur] >= left:
      left = visited[cur] + 1

    visited[cur] = right
    if right - left + 1 > len(ret):
      ret = shows[left:right + 1]

  return ret
```

### 我的反思（Coding）
- 我选择了能保证正确性的 dictionary 版本，确实不如 set 简洁；在压力下优先稳是本能，但从“代码美感/简洁性”的角度不加分。
- 面试问“还能不能更好”，我应该把 set 版也写出来（即使晚几分钟），或者先用口头把 set 的思路讲清楚再动手。

---

## 我觉得可能委屈的点 & 想请大家评价
- TPS 我确实有问、有确认，但仍然多次误解；是不是应该用更强的“Example oriented clarification”来合作（collaborate）？
- Coding 第二题我在时间压力下交付了正确、O(n)、可读的解，但不够“clean”；这种情况下该不该挂？
- 如果你是面试官，你会怎么打分？ 是不是Netflix bar很高？
# https://www.1point3acres.com/bbs/thread-1162073-1-1.html



11.
第一轮 coding
实现一个 class 支持 execute command,  command undo;
基本就是考察数据结构, 保留历史 execute command, 然后怎么undo last command


第二轮 data modeling
地里面提到过的实现 ads data, 大家可以查一查, 这轮就面的很难受了已经, 感觉根本做不完


第三轮 system design
实现ads audience targeting system, 支持大数据上传


然后是两轮 manager 的 behavior
重点是有一轮说是 domain experience, 问到麻了

# https://www.1point3acres.com/bbs/thread-1161828-1-1.html



12.
看了最近一年的面经 广告组都是拓扑 轮到我来了道timed KV cache 面完搜了一下是别的大组面经里的 难怪没写到😓


# https://www.1point3acres.com/bbs/thread-1161525-1-1.html


13.
面经没看到，难道是新题
music playlist

支持 add， remove，shuffle
bug free 一次过，要现场跑test。

歌曲有听的timestamp，add一首歌曲何时听的。
实现一个api 返回所有收听歌曲，按照时间顺序
remove一个歌曲收听记录
删除所有歌曲收听记录。
应该没有shuffle，记错了。一个月前面的。
不难，现场写都能写出来的程度


# https://www.1point3acres.com/bbs/thread-1160829-1-1.html



14.
地理面经题file system，非leetcode原题

具体题干非常宽泛 需要design一个resilient file system，确保如果system crash了能够恢复file内容，就一个描述没有code。
我是大致按照这道题 (https://leetcode.com/problems/snapshot-array/description/) 来做的。followup问了很多bottle neck和进production的话要如何改的问题

# https://www.1point3acres.com/bbs/thread-1159115-1-1.html



15.
面试的是Ads team, L5 senior
SD: Ads frequency cap
自我感觉还可以，发挥中等偏上吧，按照自己做ads的经验来说的
面试官feedback是scalability 讨论不够，communication 不够清晰
Data modeling for ads
一个南亚面试官面的我， 我体验很不好因为他自己都表达不清楚自己的要求，讲话很慢
只是说让设计ads data model, 我提了几个clarification问题，他说可以，结果讲完以后又说不是这个方向。。。我请问呢。。。
自己的感觉是被这个面试官黑了，我自己hands on做ads做了很多, 我说的有些概念他看上去并不懂，所以我很question这个人是不是真的了解ads
Coding
一个tree related problem, 没在地里见过, 题目细节不是特别记得了，跟tree getdepth有点类似，不难，需要pass test (自己写的test case)
印度小姐姐面的我，面试中很collaborative, 据说给了我好的feedback, 感谢她
BQ 1 & BQ 2
跟两个hm聊，据说都给了比较好的feedback
问了我很多lead 项目 / debug / give feedback 的细节，我确实自己做过很多事情，所以答得比较顺利，面试前也根据网飞memo 改了一些细节。

大概是两个星期后跟我说不move forward了
体感上觉得网飞的communication确实很重要，SD我是第一次准备，确实是我弱项我还需要好好提高。
我遇到的coding题整体不难，我非常active的去和面试官沟通问题，不知道这是不是让我coding round都很好的原因，我的phone screen的面试官给了我strong hire （感谢他！！！）这个面试feedback还帮我拿到了其他组的attention (但是因为背景不是很符合就没有继续了）
之前听说BQ很难过，结果这次我两轮是tech round表现不好，真是想哭TAT
心理很复杂，我知道面试实力和运气缺一不可，不过我觉得自己真的很适合这个位置也很想去（毕竟我hands on做了很多ads, 还TL了好几年，非常collaborative 还负责 )
如果有网飞的面试官，可以告诉我一下data model要怎么回答才能到点上吗？无比感谢！

# https://www.1point3acres.com/bbs/thread-1158092-1-1.html



16.
店面是一道topo的题目，面完半小时hr通知过了

vo第一轮是ads data modeling，地里的原题，lz在Meta ads干了四年了，对这方面还是比较熟悉的，周末系统的复习了一下鱿鱼厂的ads data model，自认为准备的很好。面试官是个三哥，全程在低头玩手机，我觉得反正答得很不错，应该问题不大

第二轮system design也是地里原题，ads limiter，提前用chatgpt准备了一套答案。面试官感觉很懂这方面，人挺聪明的，面试的时候各方面讨论的挺好的，当时觉得唯一没答上来的地方是让我现场写出redis和kafka的code，但是总体也感觉问题不大

第三轮coding还是topo，比店面的稍微简单一些。面试的时候早上8:30，脑子没热开，写了一两个小bug，国人小姐姐直接给我指出来了，感觉如果bar很高的话这轮应该会fail

总的来说感觉三轮technical都表现的还不错

第四轮manager面是domain and product experience啥的，狠狠准备了一下过往的各种projects以及和ads相关的project，结果面试官全程问的都是纯bq的问题，我很是迷惑，感觉没发挥的很好


最后一轮culture的manager面，我以为是纯bq，结果上来先问了我将近30分钟的纯technical的东西，问的我浑身冒汗，回答的确实不是很好 (我是做user facing product的，面试官是做ads measurement的，
	问的问题大部分都是集中在如何log data处理data那些)。切换到bq的时候面试官一直在质疑我的回答，说话的方式很passive aggressive，问的我心拔凉拔凉



面完后两轮hm面之后觉得应该是gg了，果不其然两三天后收到了not moving forward，很伤心hhh

和hr约了个followup meeting想collect some feedback，结果太出乎我意料了，hr说两轮hm面的feedback非常positive、extremely strong （这么看最后一轮的面试官可能是想pressure test interviewee？）

coding round的feedback也很好，但是system design fail了，说doesn't have a clear image of the design (感觉feedback说的比较vague我也没听懂)，modeling也fail了，说missing key component in the design

modeling那一轮我就当纯遇到沙比了，没办法，运气不好，估计从一开始就没打算给我过

system design fail了对我打击很大，难道我这么菜吗😭 我真觉得面的挺好的，ChatGPT给的答案我看起来感觉也很solid啊


有两点感受，一是Netflix的bar挺高的，二是面试真的很看运气，pass还是fail可能就是面试官一念之间

# https://www.1point3acres.com/bbs/thread-1157088-1-1.html



17.
bq什么的基本常规，data modeling地里也有

sd billing system地里很多人有提，参考 https://www.1point3acres.com/bbs/thread-1151351-1-1.html

和

https://www.1point3acres.com/bbs/thread-1151351-1-1.html

coding 是一堆用户看电影的历史里，找出最后 k 部电影的集合相同的用户分组，follow up 改成找任意两个人最后 k 部电影里有至少 m 部重叠就算一对朋友并返回所有这样的 pair。
参考：https://www.hack2hire.com/companies/netflix/coding-questions/68e2c587dab409e303d92946/practice?questionId=68e2d042dab409e303d92952


# https://www.1point3acres.com/bbs/thread-1157006-1-1.html


18.
应该这个帖子的同一个题目 https://www.1point3acres.com/bbs/thread-1155090-1-1.html

但是不确定同一个面试官

组里需要写Java，但是我太久不写java了。用java试试 被叫停 说你还是换python吧。


coding确实rusty了，后面followup design什么的没感觉有什么问题。

让我比较不爽的是时间都到了，面试官说还有时间回答问题，那我就随便问问吧，然后就疯狂夸这个组有多好多好，超时了十分钟。然后把我挂了。

# https://www.1point3acres.com/bbs/thread-1156729-1-1.html



19.
店面面的是保证synchronization 的kv store, 感觉java直接简单用concurrent hash map + atomic operation 就行了。但是我面试用python准备的，只能写了个简易的 fine grained lock.

# https://www.1point3acres.com/bbs/thread-1156112-1-1.html



20.
非 leetcode 原题, 大哥自己出的题,

大概内容就是实现一个in memory 的 文件管理系统,

支持增删改查, 以及支持 version, 难度不高


主要是一直有 follow up, 有点小 system design 的意思,

讨论在实际系统里面怎么使用,

当天出结果, 面试过程还挺好的

和leetcode in memory FS 的区别是你要搞version control，还有改动文件
leetcode版好像是append only


# https://www.1point3acres.com/bbs/thread-1155090-1-1.html




21. [MUST READ - IMPORTANT]
汇总一下目前见到过的technical 面试题

1 - 数组查重 I II III (力扣- 饿幺柒, 饿幺酒, 呃呃灵）

2 - 力扣 舞遛 （合并重叠区间）

3 - 力扣 饿灵柒 Course Schedule


4 - 实现 Linux jq 命令

5 - https://www.1point3acres.com/bbs/thread-1088915-1-1.html

6 - wighted cache - https://www.1point3acres.com/bbs/thread-1121741-1-1.html

# https://www.1point3acres.com/bbs/thread-1154747-1-1.html


22.
给个url array，每个url对应一个txt file， 里面是user review 以及一个optional 的 url指向下一个file。 
还给一个negative key word dict， 收集每个movie对应的所有negative review。

# https://www.1point3acres.com/bbs/thread-1154560-1-1.html


23.
因为reschedule先面2轮bq 都是白人大叔 感觉比较aggressive 第二个还是临时来过来的 没问几个问题就想结束了 体验很一般 而且问题还有重复 问题是 
proud project with impact，conflict with manager， constructive feedback you give/get， challenge you find and resolve

国女（abc？） data model design 重点看db schema 题目是奶飞内部的员工想要post promotion on各种平台比如fb yt tt，怎么去设计这个posting schema， 需要考虑region i18n promotion需要花的钱等等

东欧哥 SD billing system 需要支持300m subscription 重点是怎么处理recursive billing， db scan with cron job 和delay queue都ok

白人老哥 刷题网呃呃令 follow up 是O1 bucket sort 聊的还比较high


# https://www.1point3acres.com/bbs/thread-1151351-1-1.html



24.
题目不太难，是个topo sort，里扣幺幺伞流
我看面经，一直以为topo是onsite考的，店面我看他们都是考一二三组合或者kv那题，所以就没准备这个，硬着头皮上了，好在题目不难
这个时候另外一个意想不到的事情发生了，coderpad somehow没有auto syntax check，我都没import queue unordered_map那些，也没给我报错；故意写错一个variable name也还是绿的，于是我就问面试官是不是得turn on一下syntax check，他说应该已经有了，我心里emmmm没有吧？反正最后相当于整道题目就用text纯盲写的
果然不是很靠谱，写完跑test case的时候有个error说找不到一个variable declaration。我一顿debug，又是slog又是改写法的，估计得过去了十分钟也找不到问题；这时候A说你们先继续讨论吧，他来debug，他感觉是coderpad setup的问题不是代码的问题；于是B开始问我一些followup，但是我脑子里完全无法集中精神回答，一直在想bug是什么，所以这几个followup答得估计都很差；五分钟后A说他发现问题了，我loop少写了一个bracket
好吧，确实是我的typo hhh，但是如果有syntax check应该很快就能找出来.......


由于这个浪费了太多时间，followup没时间写了，我就简单说了一下解法（主要我也不知道要问两道题目，要不然我第一题就写快点了）
followup是里扣摇摇就死；
我就说了一下这种情况下每个semester怎么在available的课里面选上哪几门。面完之后问chatgpt，给的最优化的答案是bitmap+dp，这个我当时没想到

反正感觉could have done a better job if things went smooth lol

# https://www.1point3acres.com/bbs/thread-1150534-1-1.html



25.
第一轮店面 -》 实现一个可以expire的key value -〉 followup 怎么删除expired key

# https://www.1point3acres.com/bbs/thread-1149933-1-1.html



26.
利口 尔腰妻 尔腰酒 尔尔灵

领英上勾搭HM拿的面试，ML Plat组
先是HM面过了，就问了技术背景，过去project，算是半个behavior半个project
HR面过了，就花了15分钟，问了为什么奈飞，大概期望薪资

先是一个店面轮，如果过了就是VO，结果楼主自己菜得抠脚，刷题熟练度不够挂了

本身自己说第二题hashmap做，面试官说如果用set呢，然后我说那就可以sliding window吧

结果脑子卡壳了愣是没反应过来，一直想着用set的length不同，但其实membership check就够了

连第三题都没开始lol，所以我第三题是猜的但应该差不多

哎之前把第三题刷了好几遍应该够了，但结果卡在了自己认为没问题的第二题上，换了个解法脑子就卡壳了，下次努力吧

# # https://www.1point3acres.com/bbs/thread-1149933-1-1.html



27.
SD 是 ads frequency cap。跟普通rate limiter不太一样的点在于read跟write path会分开，因为有bidding的存在，
read完frequency cap并不保证ad被serve所以一般write在event tracking的那部分
Data modeling就是ad demand的部分，设置campaign啊什么的。有这部分经验的就很简单，没有的就提前问ChatGPT学就行了
Coding忘了具体题目的背景是啥了，但本质就是个topo sort
BQ就是正常问题，但感觉他们更看重有没有product sense，所以准备的例子最好有比较突出的这个点
# https://www.1point3acres.com/bbs/thread-1144843-1-1.html


28.
problem solving round， 是一个open ended problem,设计 and code 网飞主页 to  make sure主页上显示的title都是 unique
SD: 设计网飞  billing system 300M users monthly subscription billing
网飞 culture
past experience and project

电面是 力扣 三 ，这道题的各种变形，核心都是这道题

# https://www.1point3acres.com/bbs/thread-1144596-1-1.html



29.
coding sliding window 。 面试官是那种每一行都要打断你的人

吭哧着写完了 测完了
Problem solving
Netflix home page show dedupe

写了dedupe code 然后其他implementation聊了下 比如这该是前端还是后端，怎么测试，万一某一行dedupe完 不够数怎么办之类的


# https://www.1point3acres.com/bbs/thread-1144174-1-1.html



30.
LC 2622. Cache With Time Limit

Follow up：clean up expired data.
对自己很生气，有个地方写法不当（逻辑没问题！！）导致test case过不了。debug 不是很友好，当场没看出问题。下来一秒解决。。。还是面试经验不够。


# https://www.1point3acres.com/bbs/thread-1144010-1-1.html



31.
OOD design + coding

设计一个framework可以execute 有dependency的task
task interface和framework 需要自己design
太久没做Java的generic implementation所以挂了。

楼主请问一下这是ads组吗
不是

# https://www.1point3acres.com/bbs/thread-1141969-1-1.html



32.
两轮电面，一个是coding，一个是problem solving
coding就是leetcode 叁的变形，把number换成了show name或者要求不太一样
但道理是相通的
problem solving的重点不在于写code，
而是怎么和面试官沟通，
解释你的思路以及找到一个合理的方案。
题目是Netflix home page的view port rendering，
如何实现第一屏里所有行和列都没有重复的show

# https://www.1point3acres.com/bbs/thread-1139391-1-1.html



33.
题就是地里那道timebased kv store，不太难，很快写完了，聊了聊内存满了怎么办，很常规。挂了挺奇怪的，可能交流不太好，面的前几家不太在状态，上来直接写，说话太少了，
他们家hr表示题写的怎么样不是决定性因素，交流很重要。他们家面试官人不错，对我简历认真读了,表示很有兴趣，聊了些tech stack，感觉他们家可能aws，java用的挺多

# https://www.1point3acres.com/bbs/thread-1138500-1-1.html



34.
首先是代码轮，就是普通的拓扑排序。但是输入是纯字符串所以楼主花了点额外时间提取依赖关系，导致没啥followup。

广告系统设计，题目是设计ads frequency cap，相比于rate limiter而言会多一个跟广告商交接然后set up config的部件。

Data modeling就是设计广告商intake的数据库，如果是狗家的话类似于设计F1 schema。只涉及最基本的几种entities。

然后是两轮BQ，按理说应该一个侧重文化一个侧重deep dive但是我觉得两个差不多。楼主没怎么准备过BQ，感觉没答好可能会挂在这里。😭

# https://www.1point3acres.com/bbs/thread-1137301-1-1.html


35.
相同1，2，3系列
应该是利口 尔尔令系列

# https://www.1point3acres.com/bbs/thread-1135055-1-1.html


36.
SD: frequency cap. 我把我们现在组里的capping 系统直接搬上来讲了。问得很细致：Redis key 用什么，value 是什么。不知道面试官是不是想听key是userId + eventId. 
但是根据一般广告系统容量每天千万级别用户，10万的新增广告产量，我用了userId 作为key.  
coding: 很简单的topo

modeling: 和之前地里面经一样
HM轮：体验很好。能够感受到每个HM tech 背景很强。
很快收到了拒信。其实不想发面经帖子的，毕竟面试体验非常好。但是SD那轮问得有点太细致了。
除了Redis 以外，还问了capping 系统里非常细致的问题。有点怀疑他们是想知道我们公司是怎么做的。

我是onsite被HM BQ挂了 feedback是不会处理priority across team conflict HM tech很弱 感觉像是有别的选择故意挂我...

# https://www.1point3acres.com/bbs/thread-1134978-1-1.html


37.
非leetcode，要求实现一个可以expire的key value缓存。一开始实现了懒惰删除然后讨论一下怎么避免OOM，要求起另一个线程去删过期数据，
讨论了一下如果要求固定size怎么搞（就是LRU），最后讨论了一下API设计。

我觉得非leetcode这个说法有点误导. 不光LRU是leetcode, 可以expire的key value缓存也是leetcode: 981. Time Based Key-Value Store. 其实这种面试考的还是leetcode, 只是稍微披了一层大家都能看穿的不重要的皮.

# https://www.1point3acres.com/bbs/thread-1134805-1-1.html



38.
HM：project 只问了表面的部分，和一点点细节（看hm的状态，稍微一说多感觉他就想move on）
BQ问了很多，cross team/ lead project experience / project delay/ why netflix/ why next role

# https://www.1point3acres.com/bbs/thread-1130453-1-1.html


39.
月中旬recruiter reach out。5月初电面两轮，一轮coding，一轮hm。挂在hm轮。

coding: 耳药器，耳药酒，耳耳凌，当场好评。


HM: Project Deep Dive, 问了处理conflict, challenge parts, cross functional collaboration。


# https://www.1point3acres.com/bbs/thread-1130086-1-1.html



40.
一轮是coding, 给一个数组，取两个数字做加减乘除操作，输出加减乘除操作后正好是target值的所有pair。follow up是三个数字做加减乘除操作后正好是target的pairs

一轮是problem solving，从大量的邮件往来历史中找出 spam email，讨论了如何判断是spam email ，如何计算这个detect spam email 系统的准确性 写了少量的class code 感觉比较open ended

# https://www.1point3acres.com/bbs/thread-1128296-1-1.html



41.
面的数据平台org

第一轮代码
图题 但是边有点特殊 有起点终点 然后还有时间戳
实现两个接口
1 - 添加边
2 - 给定起点 终点 深度 起始时间 结束时间 返回所有的边
在实现前要先讨论怎么实现 有哪些tradeoff 比如正常的怎么存这个图 然后假如时间戳很多有1m个 怎么存能让搜索效率变快
凌鹰来的烙印



第二轮多线程
设计个计数器 实现加一和减一 还有实现个函数 多个线程可以wait这个函数 然后计数器为0的时候 这些线程全被notify
雨林来的烙印

第三轮系统设计
有个source db 有个target db 要求就是把source db的WAL log enrich一下 发给target db
write load 大概1M/s
电视软件来的南美白人


第二天面了两轮BQ 都很常规

# https://www.1point3acres.com/bbs/thread-1128181-1-1.html


42.
Implement a function timer that takes a single argument seconds whose unit is an integer representing the number of seconds. The function returns a string representing the time in a human readable form up to Months (assuming 30 days in a month). Examples:

        timer(55);      // returns: 55 seconds


        timer(65);      // returns: 1 minutes, 5 seconds

        timer(3805000); // returns: 1 months, 2 weeks, 56 minutes, 40 seconds


        // Requirements:// Don’t print zero units except for seconds - example:
        // Good: 5 minutes, 0 seconds
        // Good: 2 days, 3 minutes, 10 seconds
        // Bad:  2 days, 0 hour, 3 minutes, 10 seconds
        // Don’t use standard built-in language libraries (date, time, etc..)
        // Function to be recursive - starting off iterative to find a pattern is okay
这道题并不难，但是一开始没有看到要求用递归来写，所以花了不少时间写Iterative的code，写code比较烦。需要模除月，周，天，时，分，秒。
写完后，才被提醒要求用递归来改写。但思考再三，觉得用递归比较勉强，最后写了，但来不及运行，最后就挂了。如果一开始意识到用递归，会有比较多时间来思考，应该不会这样。
这样简单的题目挂了，觉得比较冤。

# https://www.1point3acres.com/bbs/thread-1127788-1-1.html


43.
面试形式如下，希望对大家有所帮助！
介绍HM background
介绍my background
为什么 looking?
为什么 Netflix?
Talk about a challenging project and why it is challenging? 不需要 super technical detail.
What is your approach to handle project risk and delay?
How do you handle challenging conversation?
What is your mindset to collaborate well with xFn team?
What is the difference b/w a good team and a great team? 这一问相当有意思，第一次被问到
Q & A

# https://www.1point3acres.com/bbs/thread-1127717-1-1.html


44.
Data modeling轮：设计Ads demand intaking的数据模型。有ads（尤其是measurement）背景的这一轮应该不是很难。
两轮Behavior，一些常规的bq，尽量贴着value memo答。个人认为比较难答的是“如何平衡long term和short term work"， “如何prioritize your work"
SD：地里说过的ads frequency cap，有一个follow up是假设有个对每个ads的impression cap。类似coupon系统。
coding：一个包装了的topological sort的题

# https://www.1point3acres.com/bbs/thread-1125983-1-1.html



45.
店面：rate limiter coding + rate limiter system design，hr第二天练习说白人大叔感觉聊的很好，move to onsite
vo1

onsite：sd - content recommendation system
coding：meeting room
vo2
一个senior manager的product&domain knowledge bq， 一轮hiring manager bq + role expecation, 应该是挂在bq上了，还是很难受的，但是面试体验感还是不错的。
最后一轮hm感觉面完就感觉会挂，我bq讲的太technical了，hm没有很follow，想故事讲的更完整，但是起到了反作用，也是一个教训。
另外就是我做的部分和hm组里做的东西不是特别match，虽然都是一个domain，导致沟通和理解上存在一些问题。hr说周一debrief，
周一大早上9点被拒，感觉hm都没有推进到debrief就给我挂了。

# https://www.1point3acres.com/bbs/thread-1125262-1-1.html


46.
狸扣三个月泰格题之一 銳忒厘米特

# https://www.1point3acres.com/bbs/thread-1124449-1-1.html


47.
实现weighted cache的get和put
初始设置total weight limit

达到limit后把weight最大的key-value pair删除
用treemap就可以保证get和put都是O(logN)  面试官表示赞赏 说大部分人用heap导致复杂度O(N)
# https://www.1point3acres.com/bbs/thread-1121741-1-1.html


48.
coding & problem solving都不算难，前者很像LC里的rate limiter，后者是给你一个用户界面，按user preferences排序，然后展示（返回一串节目id），什么格式都可以，挺开放的，我个人觉得奈非不是很掐算法或者最优解。
主打一个交流技吧，聊天很愉快，能感觉eng vibe挺轻松挺友善的。
# https://www.1point3acres.com/bbs/thread-1124430-1-1.html


49.
Design auto-expire cache，类似 LC 2622 。
面试的时候只有文字描述，需要跟面试官讨论具体实现什么，怎么实现。class AutoExpireCache:
    def __init__(self):
   
    def set(self, key, value, timestamp, expire_period):
   
    def get(self, key):
我写的是这些API。也讨论了input type是什么，cache miss的需要返回 None instead of False because False can be the value，expire的时候是否需要return error message和error code（不需要），能否直接override key，等等。
实现比较直给，我就用了dict。需要run code，写基本的tests（assert in Python）。
Follow up 1:
如果memory不够怎么办。
我看以前的帖子里提到了cron job和lazy delete，就和面试官讨论了一下。这里没让写code。
Follow up 2:
如果memory还不够怎么办。
我感觉面试官的意思是实现sized LRU cache。class AutoExpireCache:
    def __init__(self, k):
由于一开始用的dict，面试的时候没有反应过来，后来就没写完就到时间了。其实用个OrderedDict就行了。估计gg了。


# https://www.1point3acres.com/bbs/thread-1120276-1-1.html


50.
考了给定一个时间，比如1min，写一个print函数，input是event，有name和timestamp，要求如果在1min以内有相同的event则只print一次。

先用基本解法拿map存起来见过的event和timestamp，每次新的event进来检查上一次见这个event的时间如果间隔超过1min就输出，否则就ignore。
follow up讨论了如何删除map里过期的数据，比如用LRU，lazy delete还是cronjob定期清理。还问了如果event的timestamp是乱序的怎么维护LRU。
网飞的面试官非常看重实际情况的处理，会有很多讨论。求加米

# https://www.1point3acres.com/bbs/thread-1118557-1-1.html



51.
Coding: 叁叁贰的变种，follow up print all path
Problems solving: open ended, design movie recommendation, 挂在这轮了

# https://www.1point3acres.com/bbs/thread-1117704-1-1.html


52.
上个月的 奈飞技术店面 问的就是地里的Log rate limiter, 先讨论下本体实现是啥样 
然后开始讨论如果数据量并特别大或者这个rate limiter要运行很久会有什么问题 
然后讨论了一下怎么删除数据 然后又实现了一下数据删除的部分 
挺实际的问题 面之前特地看了下rate limiter的system design 很有帮助

# https://www.1point3acres.com/bbs/thread-1117598-1-1.html


53.
round 1: k closest distance, leetcode 原题
round 2: design ML job scheduler，没有任何这方面的经验。。。只能乱答，面到最后面试官都无语了

round 3: home page video recommendation，花了更多时间讨论需要多少个service，没怎么讨论feature / modeling / evaluation相关的问题（一笔带过）

# https://www.1point3acres.com/bbs/thread-1110513-1-1.html


54.
题目很简单，找重复数字，一共三问
面试官很好很赞，但是CodeSignal somehow各种bug，面试过程各种不顺，莫名就挂了


L1: 给定一系列观看ID, 如果有重复观看，返回True，不然False e.g. [4, 5, 100, 200, 5] -> True
L2: 如果只在最近的K window内有重复观看，返回True，就是一个sliding window
L3: 给定一个ID range T，如果两个ID距离小于等于T，认为属于一个系列，返回True，比如T=4，[1, 5, 100]-> True，但是T=3的话返回False

# https://www.1point3acres.com/bbs/thread-1107464-1-1.html



55.
SDE 60 min
这轮应该是算法。但是没有直接给一道题，更像是OOD。前15分钟互相介绍，对面介绍了不少组里的东西。中间30分钟开始写题。
给的场景是有一套输入，类似 {type: email, value: xxx@xx.com}的list, 也会有内嵌 {type: contact, value: {type: email, value: xxx@xxx.com}}. 
让我design一个java class来保存这个数据，然后写个函数返回第一个没有重复的有效email。大概花了几分钟去和他确认input的结构，这个内嵌会有多少层，
type的值是不是可枚举，等等。因为他的description就直接给了好多sample的json，我第一反应就是这个java class的构造函数的参数就是一个json字符串。
开始想怎么parse，边想边给他解释我要怎么去一层层去parse然后直到发现type：email。他发现我的方向有点歪，
给我说不需要parse string，给的input只是sample。确认了半天，原来是说我只需要写简单的构造函数，接受type和value，具体的嵌套是一层层new出来的。
这会已经用了快十分钟了，我只能赶紧的给出个简单的类似Node，里面有个value也是Node，同时Node也有个函数getEmail。
然后快速的写了找第一个不重复的email。 最后开始写unit test。这就是最坑的地方了，sample input已经给了，
大概要new五六个instance，还要设置内嵌关系。我只能不断的复制粘贴，node1，node2，node3.。。。中间一度的在想要不要写个util函数。
最后写完了，运行，结果也是预期的。但是他觉得不对劲，让我再写个test case，用另外的sample input。这个sample更长。。。
我画了更多的时间new完一堆的nodes，跑了下，结果不是预期的。我看了下我的输出，发现输出的email是个无效的email。
这才想起还有个validation过程。。赶紧说怎么加validation，一分钟之内在加好，跑完，结果预期。
但是整个时间已经很赶了。最后只能口头上说怎么优化，大概就是根据type抽象成不同的类型，然后用点Factory方便生成instance。接着也有几分钟QA

# https://www.1point3acres.com/bbs/thread-1107177-1-1.html


56.
两周前面的，形式有些特别，share screen给看了个demo UI，实际就是蠡口意思刘。


花了很久讨论怎么定义API，加文档，方便前端实现 UI，导致写代码时间紧张，差点没写完。

# https://www.1point3acres.com/bbs/thread-1083505-1-1.html

不是最优的，最优的是用个list

# https://www.1point3acres.com/bbs/thread-1101812-1-1.html



57.
电面：一位年轻的senior 小哥，加上东欧大叔shadow. 
很free-form：implement rpc rate limiter，没有什么具体要求；有点像是把sys design 和basic coding结合。
简单实现一下后，开始讨论各种scalability concerns和quota management design。 很practical，挺有意思的。


hm 面：very basic behavioral problem, nothing special; 看一下netflix memo


onsite #1 part1: 很友好的老印大叔：讨论过去的oncall/triage 经历，深入讨论mitigation细节
onsite #1 part2: 设计一个remote OS rollout system. CAVEAT: 系统将会被部署在深海游轮上，with very slow and unreliable bandwidth, and can't be physically maintained until get to the shore (O(years)). 非常有意思，讨论各种corner case， changes needed due to restraints，以及怎么simplify。


onsite #2：友好的老印小哥：设计一个service (i.e. tasks listening on Http port on linux servers) discovery system (类似于stubby) for multi-tenant compute platform, within a datacenter. 这里有具体的scalability计算：xxk servers, each with m containers, and each with n different types of tasks. 很注意practical implementation，底层设计，以及各种trade off。


onsite #3：东欧大叔: 实现一个crawler/parser来收集在很多servers上的log message。前半部分实现，后半部分讨论alternatives/scalability。


onsite #4：hm #2： 很普通的行为问题。还是比较深挖细节，所以得提前整理一下。


# https://www.1point3acres.com/bbs/thread-1097086-1-1.html


58.
tech店面：LC司仪

SD轮
netflix cloud gaming
Component: 排行榜
Follow up: design interface for developer


# https://www.1point3acres.com/bbs/thread-1094593-1-1.html


59.
r1: 高并发：设计一个 class 需要保证多个 user 想你report 他们的latency，然后他们想知道一段时间的 99th 的latency 是多少

r2：coding 给一个 json style，和 jq path 找符合 jq path 的value
r3:  sd：设计一个 cron system，follow up: 高并发的时候 分库和分别，多server的时候如何 allocate server 和 table 的关系，如何知道 server down 了，以及server down了之后怎么办，因为我用了 cluster的思路，所以说再问了 master node down了怎么办，如果 master node 短时间无法重启，worker node 也down了怎么办


# https://www.1point3acres.com/bbs/thread-1089519-1-1.html



60.
这个题有点像lru的变种， 考同日志不输出，如果log一样的话，不要在短时间内输出(会给一个时间range)。之后会有很多follow up, 类似于怎么删除数据，怎么处理concurrency以及有什么strategy去实现最优的删除策略

# https://www.1point3acres.com/bbs/thread-1088915-1-1.html


61.
面的是一个ad的组

店面：力扣636

VO：
System Design：设计一个推广ad的System，define db schema， 要多关注他们的系统
力扣 56， 还有一个是自己出的题。

# https://www.1point3acres.com/bbs/thread-1087707-1-1.html


62.
coding 1: interval intersection 变种,找出做多overlap的intervals个数。follow up1 找出给定时间点有多少overlap。follow up2 时间点变成参数，找出任意给定时间点的overlap数量，不用写出来。


Coding 2: each customer has list of movies, if 2 customers have common last k movies, then they are friends. find all friends. follow up  if m out of last k movies are common, then friends.

# https://www.1point3acres.com/bbs/thread-1085665-1-1.html



63.
许多用户每天都会访问 Netflix 网站。假设我们跟踪客户正在观看的视频，用于业务指标等。

每次有人访问网站时，我们都会将记录写入日志文件，其中包含时间戳、视频 ID、客户 ID。一天结束时，我们会得到一个包含许多该格式条目的大型日志文件。而且每天都会有一个新文件。


现在，给定两个日志文件（第 1 天的日志文件和第 2 天的日志文件），我们希望生成一个符合以下条件的“忠实客户”列表：(a) 他们两天都来过，(b) 他们观看了至少两个独特的视频。

following: 文件特别大的时候要怎么办



# https://www.1point3acres.com/bbs/thread-1084816-1-1.html







64.
BQ

1. Past projects - most challenging part, your role

2. Tell me a time when you disagree with the team
3. Tell me a time when you inherited a system in bad shape
4. How do you prioritize tasks

# https://www.1point3acres.com/bbs/thread-975350-1-1.html


65.
HM:
Talk about a project you led where you knew it wasn't going to succeed.'
Talk about a conflict you had with a coworker.
Talk about a conflict you had with your manager.
Talk about your favorite project.
What do you want in your next role.

# https://www.1point3acres.com/bbs/thread-953945-1-1.html


66.
given and received feedback, pivoting(pivot fast), ownership, project with impact， 
conflict, ambiguity, how do you define success project outcome


biggest mistake you made


Culture
How do you handle conflict and challenge



proud project with impact，conflict with manager， constructive feedback you give/get， challenge you find and resolve



feedback是不会处理priority across team conflict HM tech很弱 感觉像是有别的选择故意挂我




cross team/ lead project experience / project delay/ why netflix/ why next role



如何平衡long term和short term work"， “如何prioritize your work"





































                                      +----------------------------------+
                                      | Advertiser / Ad Ops              |
                                      | - campaign config                |
                                      | - budgets / pacing policy        |
                                      | - frequency cap rules            |
                                      +----------------+-----------------+
                                                       |
                                                       v
                                      +----------------------------------+
                                      | Config / Policy Service          |
                                      +----------------+-----------------+
                                                       |
                             +-------------------------+--------------------------+
                             |                                                    |
                             v                                                    v
              +-------------------------------+                 +-------------------------------+
              | Config DB                     |                 | Config Cache                  |
              | campaign metadata, rules      |                 | local + distributed           |
              +-------------------------------+                 +-------------------------------+


Viewer request / ad slot opens
           |
           v
+-----------------------------+
| Core Ad Decision Engine     |
| (ad pod / serving frontend) |
+-------------+---------------+
              |
              | request context:
              | profile_id, geo, device, content, etc.
              v
+-----------------------------+
| Candidate Retrieval /       |
| Targeting Service           |
| - inverted index / bitmaps  |
| - returns candidate ads     |
+-------------+---------------+
              |
              | candidate campaigns / creatives
              v
+----------------------------------------------------------------------------------+
| Bidder Fleet                                                                      |
| (many bidder processes across regions)                                            |
|                                                                                   |
|   +----------------------+   +----------------------+   +----------------------+  |
|   | Bidder Shard A       |   | Bidder Shard B       |   | Bidder Shard C       |  |
|   | - pre-rank           |   | - pre-rank           |   | - pre-rank           |  |
|   | - local cache        |   | - local cache        |   | - local cache        |  |
|   +----------+-----------+   +----------+-----------+   +----------+-----------+  |
|              |                          |                          |              |
+--------------+--------------------------+--------------------------+--------------+
               |                          |                          |
               | batch checks on reduced candidate set               |
               v
+----------------------------------------------------------------------------------+
| Real-time Constraint Services                                                    |
|                                                                                  |
|   +----------------------------------+      +----------------------------------+ |
|   | Budget / Pacing Service          |      | Exposure Control Service         | |
|   | - campaign spend check           |      | - frequency cap                  | |
|   | - regional allowance             |      | - recent brand exposure          | |
|   | - shard quota / reservation      |      |                                  | |
|   +----------------+-----------------+      +----------------+-----------------+ |
+--------------------|------------------------------------------|-------------------+
                     |                                          |
                     v                                          v
      +-------------------------------+          +-------------------------------+
      | Budget Online Store           |          | Exposure Online Store         |
      | - spend counters              |          | - profile/ad hour/day/month   |
      | - pacing allowances           |          | - profile/campaign counters   |
      | - shard local quota           |          | - profile/brand recent state  |
      | (Redis / KV / atomic dec)     |          | (Redis / KV / TTL counters)   |
      +-------------------------------+          +-------------------------------+

               ^                                                                  |
               |                                                                  |
               +---------------------- decision results ---------------------------+
                                      allow / throttle / block

After winner chosen and ad actually renders:
                                      |
                                      v
                            +---------------------------+
                            | Event Collector           |
                            | impression / billing /    |
                            | completion / no-bid logs  |
                            +-------------+-------------+
                                          |
                                          v
                            +---------------------------+
                            | Durable Event Log         |
                            | Kafka / Pulsar            |
                            +-------------+-------------+
                                          |
                    +---------------------+----------------------+
                    |                                            |
                    v                                            v
        +-----------------------------+            +-----------------------------+
        | Stream Processors           |            | Raw Event Archive           |
        | - dedupe                    |            | object store / warehouse    |
        | - update spend state        |            | replay / analytics / audit  |
        | - update freq cap state     |            +-----------------------------+
        | - update brand exposure     |
        +-------------+---------------+
                      |
          +-----------+--------------------------+
          |                                      |
          v                                      v
+-------------------------------+    +-------------------------------+
| Budget Online Store           |    | Exposure Online Store         |
+-------------------------------+    +-------------------------------+





=============================================================================



                                      +----------------------------------+
                                      | Advertiser / Ad Ops              |
                                      | creates campaigns, budgets,      |
                                      | pacing, frequency caps           |
                                      +----------------+-----------------+
                                                       |
                                                       v
                                      +----------------------------------+
                                      | Config Service                   |
                                      | validates config and writes      |
                                      | metadata/config truth            |
                                      +----------------+-----------------+
                                                       |
                                                       v
                                      +----------------------------------+
                                      | Metadata DB                      |
                                      | durable campaign config,         |
                                      | cap profiles, targeting metadata |
                                      +----------------+-----------------+
                                                       |
                             +-------------------------+-------------------------+
                             |                                                   |
                             v                                                   v
              +-------------------------------+                 +-------------------------------+
              | Config Cache                  |                 | Targeting Index Builder       |
              | hot cached config             |                 | builds serving indexes        |
              +-------------------------------+                 +---------------+---------------+
                                                                                 |
                                                                                 v
                                                              +----------------------------------+
                                                              | Targeting Index / Retrieval      |
                                                              | inverted index / bitmaps         |
                                                              +----------------------------------+


Viewer playback hits ad slot
           |
           v
+-----------------------------+
| Publisher / Playback /      |
| Ad Decision Frontend        |
| knows profile, device, slot |
+-------------+---------------+
              |
              v
+-----------------------------+
| Candidate Retrieval         |
| fetch candidate ads         |
+-------------+---------------+
              |
              v
+----------------------------------------------------------------------------------+
| Bidder Fleet                                                                      |
| many serving pods/cells; each shard has multiple bidder workers                   |
|                                                                                   |
|   +---------------------------+    +---------------------------+                  |
|   | Bidder Shard / Serving    |    | Bidder Shard / Serving    |                  |
|   | Cell S1                   |    | Cell S2                   |                  |
|   | - local allocator         |    | - local allocator         |                  |
|   | - bidder workers          |    | - bidder workers          |                  |
|   +-------------+-------------+    +-------------+-------------+                  |
|                 |                                |                                |
+-----------------+--------------------------------+--------------------------------+
                  |
                  | batch constraint checks on reduced candidate set
                  v
+----------------------------------------------------------------------------------+
| Real-time Constraint Services                                                    |
|                                                                                  |
|  +--------------------------------+     +--------------------------------------+ |
|  | Budget / Pacing Service        |     | Exposure Control Service             | |
|  | checks shard/region allowance  |     | checks frequency cap / brand rules   | |
|  +----------------+---------------+     +----------------+---------------------+ |
+-------------------|---------------------------------------|----------------------+
                    |                                       |
                    v                                       v
   +----------------------------------+      +----------------------------------+
   | Regional Allocator Service       |      | Exposure Online Store            |
   | service managing region quota    |      | Redis/Aerospike style counters   |
   +----------------+-----------------+      | per profile/ad/brand windows     |
                    |                        +----------------------------------+
                    v
   +----------------------------------+
   | Regional Allocator Redis         |
   | hot regional leaseable budget    |
   | per campaign / region / slot     |
   +----------------------------------+


If bidder wins publisher-side auction and ad actually starts rendering:
                    |
                    v
+----------------------------------+
| Publisher / Playback emits       |
| impression_started / billable    |
| event to Event Collector         |
+----------------+-----------------+
                 |
                 v
+----------------------------------+
| Event Collector                  |
| validates and appends events     |
+----------------+-----------------+
                 |
                 v
+----------------------------------+
| Durable Event Log                |
| Kafka / Pulsar                   |
+----------------+-----------------+
                 |
      +----------+-----------------------------+
      |                                        |
      v                                        v
+------------------------------+    +----------------------------------+
| Stream Processors            |    | Raw Event Archive / Data Lake    |
| - dedupe                     |    | replay / analytics / audit       |
| - update accounting truth    |    +----------------------------------+
| - update exposure counters   |
+--------------+---------------+
               |
      +--------+------------------------------+
      |                                       |
      v                                       v
+------------------------------+   +----------------------------------+
| Accounting Store             |   | Exposure Online Store            |
| durable actual spend /       |   | hot profile/ad/brand state       |
| impressions / delivery truth |   +----------------------------------+
+--------------+---------------+
               ^
               |
               |
+------------------------------+
| Pacing Controller            |
| distributed control-plane    |
| reads config + actuals,      |
| computes regional allocations|
+--------------+---------------+
               |
               v
+------------------------------+
| Pacing Control Store         |
| persistent control decisions |
| / latest allocation plans    |
+------------------------------+






















































































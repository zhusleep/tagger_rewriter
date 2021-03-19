# 抽取式多轮对话改写

## 模型结构
![模型结构](https://github.com/zhusleep/tagger_rewriter/blob/master/model.jpg)

## 运行方式
```
pip install -r requirements.txt
cd src
python3 rewrite_tagger.py
```

## 环境
```
预训练语言模型rb3,可以在https://github.com/ymcui/Chinese-BERT-wwm下载。或者修改成你自己的预训练模型
rouge==1.0.0
tokenizers==0.9.3
torch==1.7.1+cu101
torchaudio==0.7.2
torchvision==0.8.2+cu101
tqdm==4.56.0
transformers==3.5.1
pandas==1.1.5
注意 pytorch 根据自己的cuda版本选择安装。
```

## 结果
* 1 epoch
* {'rouge-1': {'f': 0.89, 'p': 0.94, 'r': 0.87}, 'rouge-2': {'f': 0.7824, 'p': 0.821, 'r': 0.7667}, 'rouge-l': {'f': 0.848, 'p': 0.890, 'r': 0.828}, 'em': 0.5}
```
------------
你知道板泉井水吗  |  知道  |  她是歌手  |  板泉井水是歌手  |  板泉井水是歌手
乌龙茶  |  乌龙茶好喝吗  |  嗯好喝  |  嗯乌龙茶好喝  |  嗯乌龙茶好喝
武林外传  |  超爱武林外传的  |  它的导演是谁  |  武林外传的导演是谁  |  武林外传的导演是谁
李文雯你爱我吗  |  李文雯是哪位啊  |  她是我女朋友  |  李文雯是我女朋友  |  李文雯是我女朋友
舒马赫  |  舒马赫看球了么  |  看了  |  舒马赫看了  |  舒马赫看球了
徐彬我好想你  |  谁是徐斌  |  他是经济学博士现在为首都经济贸易大学劳动经济学院人才系主任  |  徐彬是经济学博士现在为首都经济贸易大学劳动经济学院人才系主任  |  徐斌是经济学博士现在为首都经济贸易大学劳动经济学院人才系主任
```

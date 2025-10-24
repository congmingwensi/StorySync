calling_vits_ui是阅读器软件本体

get_image是 根据语句，转成生图prompt，再调用本地stable diffution代码。可以自行调整

test_indextts是用flask写的 index tts接口。

index-tts会很慢，跟不上朗读速度。 我自己是用Bert-Vist2的，但是接口和其他代码连在一起懒得拆了。有空了拆一版发出来。

运行方式：直接python calling_vits_ui

需要环境准备：本地文字转语音接口，本地stable diffution接口（秋叶大佬的一键包运行后自动带接口访问）。
可以自行修改自己的tts和生图接口。
朗读效果如视频：https://youtu.be/uDDNwLBCtwQ

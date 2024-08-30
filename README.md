# conversation-robot-for-open-campus
A robot system that talks to high school students visiting an open campus.


本プログラムはオープンキャンパス用の会話ロボットシステムのプログラムです。
スタッフが、参加者の氏名と生年月日をキーボードから手入力することで、カードリーダーによる個人識別を代替します。
（氏名と生年月日だけでは同姓同名かつ同じ誕生日がありえますが、他に入力しやすいユニークな情報がないのでこのようにしています）

Waitステートでキーボード入力できます。

app_mic.pyはPCのマイクを使用するプログラムです。

app_stream.pyはSotaのインテリマイクを使用するプログラムです。実行前にSotaからこのプログラムにffmpegで音声データを送信する必要があります（以下のコード参照）。
```bash
./ffmpeg -channels 1 -f alsa -thread_queue_size 8192 -i hw:2 -preset ultrafast -tune zerolatency -ac 1 -c:a pcm_s16le -ar 16000 -f s16le udp://<このプログラムを実行しているPCのIP>:5001?pkt_size=1024
```
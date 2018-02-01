# Emotion Detection from Messor

Off-time and real-time face detection and emotion detection based on fer2013/IMDB datasets with openCV andk eras CNN model.

## Instructions

Python 3.6+ is required,

```
pip3 install -r requirements.txt
```

### Sophia Demo

Download video data save it in `images` directory 

http://data.kchen.cc/video/mp4/sophia_full.mp4?attname=&e=1517477892&token=Av5Wu62pXuu3fCK2gUUdxvbtpGZaXB5Ye1-JdTQ2:M7rD7hNUlyPZXPcTj7FAqg7sRZQ

and run:

```
python3 src/sophia_video_emotion_color_demo.py
```

### Real-time Demo

```
python3 src/video_emotion_color_demo.py
```

### Run With Web

```
python3 src/web.py
```

### Show Ststistic 

Download processed video, save it in `images` directory

http://data.kchen.cc/video/mp4/sophia_processed.mp4?attname=&e=1517477892&token=Av5Wu62pXuu3fCK2gUUdxvbtpGZaXB5Ye1-JdTQ2:yP5MSMrAlXLOrdY4Ly0tATh7OvY

and open:

```
src/web/statistics.html
```
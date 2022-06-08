import cv2
from ffpyplayer.player import MediaPlayer
video_path=r"D:/face.mp4"
def PlayVideo(video_path):
    video=cv2.VideoCapture(video_path)
    player = MediaPlayer(video_path)
    ret, frame = video.read()
    while ret:
        ret, frame=video.read()
        audio_frame, val = player.get_frame()
        if not ret:
            print("End of video")
            break
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
        if val != 'eof' and audio_frame is not None:
            #audio
            # img, t = audio_frame
            pass
        cv2.imshow("Video", frame)

    video.release()
    cv2.destroyAllWindows()

PlayVideo(video_path)

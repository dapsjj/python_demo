import eyed3
audiofile = eyed3.load(r"C:/Users/Administrator/Desktop/audio [vocals].mp3")
audiofile.tag.artist = "标题"
audiofile.tag.title = ""
audiofile.tag.save()

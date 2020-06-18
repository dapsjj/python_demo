import csv

newTxtList = []
txtpath = r'E:\test_opencv\tensorflow-yolov3-master_for_12_garbage\data\dataset\train_10kinds.txt' #原有的大的txt

with open(txtpath, 'r', encoding='utf-8') as f1,open(r'D:/aaa.txt', 'r', encoding='utf-8') as f2: #aaa.txt部分图片被了新标签，结构与train_10kinds.txt类似
    f1_lines = f1.readlines()
    f2_lines = f2.readlines()
    f1_list = [ row.split(' ') for row in f1_lines]
    f2_list = [ row.split(' ') for row in f2_lines]
    img1Name = [x[0] for x in f1_list]
    img2Name = [x[0] for x in f2_list]
    for row in f1_list: #循环train_10kinds.txt每一行，用aaa.txt中的信息替换train_10kinds.txt中的信息
        if row[0] not in img2Name:  # 没有这个jpg
            newTxtList.append(row)
        else:  # 有这个jpg
            findIndex = img2Name.index(row[0])
            newData = f2_list[findIndex]
            newTxtList.append(newData)

    for row in f2_list: #train_10kinds.txt没有的图片要追加到newTxtList中
        if row[0] not in img1Name:  # 没有这个jpg
            newTxtList.append(row)

all=[]
for row in newTxtList:
    lineList=[]
    for line in row:
        thisStr = line.replace('\n','')
        lineList.append(thisStr)
    all.append(lineList)

with open(r'D:/bbb.txt', 'w', newline='', encoding='utf-8') as f: #生成的这个bbb.txt替换原有的train_10kinds.txt
    writer = csv.writer(f,delimiter=' ')
    writer.writerows(all)


from PIL import Image
arg = 'train'

if arg == 'train':
    gtfile = r"C:/Users/eseen_new/Desktop/wider_face_split/wider_face_split/wider_face_train_bbx_gt.txt"
    im_folder = r"C:/Users/eseen_new/Desktop/WIDER_train/WIDER_train/images/"
    xmlflie = r"C:/Users/eseen_new/Desktop/xml/"
    NoLabelXmlPath = r"C:/Users/eseen_new/Desktop/nolabel_xml/"


with open(gtfile, "r") as gt:
    jpg_num = 0
    picWithoutLabel = 0
    picWithoutLabelName = [] * 100

    while (True):
        # gt_con = gt.readline()[:-1]
        gt_con = gt.readline()
        print(gt_con)
        # print(gt_con.split('.')[-1],type(gt_con.split('.')[-1]))
        if gt_con.split('.')[-1] == 'jpg\n':
            print('aaaaa')
            jpg_num += 1
            print('jpg_num:', jpg_num)
            imgPath = im_folder + gt_con.split('.')[-2] + '.jpg'
            # gt_con.split('.')[-2] + '.jpg'  0--Parade/0_Parade_marchingband_1_849.jpg
            img = Image.open(imgPath)
            imgWidth, imgHeight = img.size
            print(imgWidth, imgHeight)
            bbox_num = gt.readline()
            print('bbox_num:', bbox_num)

            xml_file = open(xmlflie + gt_con.split('.')[-2].split('/')[1] + '.xml', 'w')

            if int(bbox_num) == 0:  # 数据集中存在没有标注的图，把他们找出来
                picWithoutLabel += 1
                print('picWithoutLabel:', picWithoutLabel, gt_con.split('.')[-2] + '.jpg')
                picWithoutLabelName.append(gt_con.split('.')[-2] + '.jpg')
                print('picWithoutLabelName:', picWithoutLabelName)

                bbox_num = 1
                xml_file = open(NoLabelXmlPath + gt_con.split('.')[-2].split('/')[1] + '.xml', 'w')

            xml_file.write('<annotation>\n')
            xml_file.write('    <folder>' + 'wider_face/' + im_folder + '</folder>\n')
            xml_file.write('    <filename>' + gt_con.split('.')[-2] + '.jpg' + '</filename>\n')
            xml_file.write('    <size>\n')
            xml_file.write('        <width>' + str(imgWidth) + '</width>\n')
            xml_file.write('        <height>' + str(imgHeight) + '</height>\n')
            xml_file.write('        <depth>' + str(3) + '</depth>\n')
            xml_file.write('    </size>\n')

            for i in range(int(bbox_num)):
                bbox_mess = [] * 10
                bbox_mess = gt.readline().split(' ')
                print('bbox_mess:', bbox_mess)
                x, y, w, h = bbox_mess[0:4]

                xml_file.write('    <object>\n')
                xml_file.write('        <name>' + 'face' + '</name>\n')
                xml_file.write('        <bndbox>\n')
                xml_file.write('            <xmin>' + str(x) + '</xmin>\n')
                xml_file.write('            <ymin>' + str(y) + '</ymin>\n')
                xml_file.write('            <xmax>' + str(int(x) + int(w)) + '</xmax>\n')
                xml_file.write('            <ymax>' + str(int(y) + int(h)) + '</ymax>\n')
                xml_file.write('        </bndbox>\n')
                xml_file.write('    </object>\n')

            xml_file.write('</annotation>\n')
            xml_file.close()

        else:
            print('bbbbb')
            break
